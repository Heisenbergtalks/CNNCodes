import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math, os

# Mixed precision and optional multi-GPU
tf.keras.mixed_precision.set_global_policy("mixed_float16")
strategy = tf.distribute.MirroredStrategy() if len(tf.config.list_physical_devices('GPU')) > 1 else tf.distribute.get_strategy()

NUM_CLASSES = 10
BATCH = 256
EPOCHS = 30
BASE_LR = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.1
IMAGE_SIZE = 32

# --------------------------
# Data: CIFAR-10 + tf.data
# --------------------------
(x_tr, y_tr), (x_te, y_te) = keras.datasets.cifar10.load_data()
y_tr = y_tr.squeeze().astype("int32")
y_te = y_te.squeeze().astype("int32")

def normalize(x, y):
    x = tf.image.convert_image_dtype(x, tf.float32)
    mean = tf.constant([0.4914, 0.4822, 0.4465])
    std  = tf.constant([0.2023, 0.1994, 0.2010])
    x = (x - mean) / std
    return x, y

# RandAug-like light policy
def augment(x, y):
    x = tf.image.random_flip_left_right(x)
    # pad and random crop
    x = tf.image.resize_with_crop_or_pad(x, IMAGE_SIZE + 4, IMAGE_SIZE + 4)
    x = tf.image.random_crop(x, (IMAGE_SIZE, IMAGE_SIZE, 3))
    return x, y

AUTO = tf.data.AUTOTUNE
train_ds = (
    tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
      .shuffle(20_000)
      .map(augment, num_parallel_calls=AUTO)
      .map(normalize, num_parallel_calls=AUTO)
      .batch(BATCH, drop_remainder=True)
      .prefetch(AUTO)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((x_te, y_te))
      .map(normalize, num_parallel_calls=AUTO)
      .batch(BATCH)
      .prefetch(AUTO)
)

# --------------------------
# MixUp
# --------------------------
@tf.function
def mixup(x, y, alpha=0.2):
    if alpha <= 0.0:
        return x, y
    lam = tf.compat.v1.distributions.Beta(alpha, alpha).sample([tf.shape(x)[0]])
    lam_x = tf.reshape(lam, [-1, 1, 1, 1])
    lam_y = tf.reshape(lam, [-1, 1])
    index = tf.random.shuffle(tf.range(tf.shape(x)[0]))
    x2 = tf.gather(x, index)
    y2 = tf.gather(y, index)
    x = x * lam_x + x2 * (1.0 - lam_x)
    y = tf.one_hot(y, NUM_CLASSES)
    y2 = tf.one_hot(y2, NUM_CLASSES)
    y = y * lam_y + y2 * (1.0 - lam_y)
    return x, y

# --------------------------
# Model: ResNet-style
# --------------------------
def conv_bn(x, filters, kernel, stride=1):
    x = layers.Conv2D(filters, kernel, stride, padding="same", use_bias=False,
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    return x

def residual_block(x, filters, stride=1, bottleneck=False):
    shortcut = x
    if bottleneck:
        # Bottleneck: 1x1 -> 3x3 -> 1x1
        x = conv_bn(x, filters, 1, stride)
        x = layers.Activation("relu")(x)
        x = conv_bn(x, filters, 3, 1)
        x = layers.Activation("relu")(x)
        x = conv_bn(x, filters * 4, 1, 1)
        out_filters = filters * 4
    else:
        # Basic: 3x3 -> 3x3
        x = conv_bn(x, filters, 3, stride)
        x = layers.Activation("relu")(x)
        x = conv_bn(x, filters, 3, 1)
        out_filters = filters

    # Projection if shape mismatch
    if shortcut.shape[-1] != out_filters or stride != 1:
        shortcut = layers.Conv2D(out_filters, 1, stride, padding="same", use_bias=False,
                                 kernel_initializer="he_normal")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x

def make_stage(x, filters, blocks, stride, bottleneck=False):
    x = residual_block(x, filters, stride=stride, bottleneck=bottleneck)
    for _ in range(blocks - 1):
        x = residual_block(x, filters, stride=1, bottleneck=bottleneck)
    return x

def build_model():
    inputs = layers.Input((IMAGE_SIZE, IMAGE_SIZE, 3))
    x = conv_bn(inputs, 64, 3, 1)
    x = layers.Activation("relu")(x)
    x = make_stage(x, 64,  2, stride=1, bottleneck=False)
    x = make_stage(x, 128, 2, stride=2, bottleneck=False)
    x = make_stage(x, 256, 2, stride=2, bottleneck=False)
    x = make_stage(x, 512, 2, stride=2, bottleneck=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(NUM_CLASSES, dtype="float32")(x)  # logits in float32
    return keras.Model(inputs, x, name="ResNetSmall")

with strategy.scope():
    model = build_model()

# --------------------------
# Optimizer, schedules, EMA
# --------------------------
steps_per_epoch = math.floor(len(x_tr) / BATCH)
total_steps = steps_per_epoch * EPOCHS

lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=BASE_LR,
    decay_steps=total_steps,
    alpha=0.1,
)

optimizer = tfa_opt = None
try:
    import tensorflow_addons as tfa
    tfa_opt = tfa.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY, epsilon=1e-8)
except Exception:
    # Fallback to Adam with manual L2 via losses
    tfa_opt = keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)

ema = keras.optimizers.schedules.ExponentialMovingAverage(0.999)

# Loss
ce = keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=LABEL_SMOOTH)
acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

# --------------------------
# Training step
# --------------------------
@tf.function
def train_step(images, labels):
    # MixUp
    images, soft_labels = mixup(images, labels, alpha=0.2)
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        # If AdamW not available, add manual L2 on kernels
        if isinstance(tfa_opt, keras.optimizers.Adam):
            l2 = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables if "kernel" in v.name]) * WEIGHT_DECAY
        else:
            l2 = 0.0
        loss = ce(soft_labels, logits) + l2
    grads = tape.gradient(loss, model.trainable_variables)
    # Gradient clipping
    grads = [tf.clip_by_global_norm(g, 5.0)[0] if g is not None else None for g in grads]
    tfa_opt.apply_gradients(zip(grads, model.trainable_variables))
    ema.apply(model.trainable_variables)
    # For accuracy, use hard labels against logits
    acc_metric.update_state(labels, logits)
    return loss

@tf.function
def test_step(images, labels):
    logits = model(images, training=False)
    val_acc_metric.update_state(labels, logits)

# --------------------------
# Training loop
# --------------------------
logdir = "logs/cifar_resnet"
ckpt_path = "ckpts/cifar_resnet.ckpt"
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
ckpt = tf.train.Checkpoint(model=model, opt=tfa_opt)
manager = tf.train.CheckpointManager(ckpt, directory=os.path.dirname(ckpt_path), max_to_keep=3)
tb = tf.summary.create_file_writer(logdir)

best_val = 0.0
global_step = 0

for epoch in range(1, EPOCHS + 1):
    acc_metric.reset_states()
    for images, labels in train_ds:
        loss = train_step(images, labels)
        global_step += 1
        if global_step % 50 == 0:
            with tb.as_default():
                tf.summary.scalar("train/loss", loss, step=global_step)
                tf.summary.scalar("train/lr", lr_schedule(global_step), step=global_step)

    train_acc = acc_metric.result().numpy()

    # Eval with EMA weights swap
    original_vars = [v.read_value() for v in model.trainable_variables]
    for v in model.trainable_variables:
        v.assign(ema.average(v))

    val_acc_metric.reset_states()
    for images, labels in test_ds:
        test_step(images, labels)
    val_acc = val_acc_metric.result().numpy()

    # Restore original weights after eval
    for var, orig in zip(model.trainable_variables, original_vars):
        var.assign(orig)

    with tb.as_default():
        tf.summary.scalar("eval/acc", val_acc, step=global_step)
        tf.summary.scalar("train/acc", train_acc, step=global_step)

    if val_acc > best_val:
        best_val = val_acc
        manager.save()

    print(f"epoch {epoch:02d}  train_acc {train_acc:.4f}  val_acc {val_acc:.4f}  best {best_val:.4f}")

print("Best val acc:", round(best_val, 4))
