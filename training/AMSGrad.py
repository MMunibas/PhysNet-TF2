"""
AMSGrad for TensorFlow (TF2.x compatible in TF1-compat mode)
Original author: Junho Kim
Updated for TensorFlow 2.x by ChatGPT
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class AMSGrad(tf.train.Optimizer):
    """AMSGrad optimizer compatible with TensorFlow 2.x in TF1 mode."""

    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.99,
                 epsilon=1e-3,
                 use_locking=False,
                 name="AMSGrad"):
        super(AMSGrad, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        # Tensor versions created in _prepare
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

        # Power accumulators created in _create_slots
        self._beta1_power = None
        self._beta2_power = None

    # ------------------------------------------------------------------
    # Slot creation
    # ------------------------------------------------------------------
    def _create_slots(self, var_list):
        first_var = min(var_list, key=lambda x: x.name)
        create_new = self._beta1_power is None

        if create_new:
            with tf.device(first_var.device):
                self._beta1_power = tf.get_variable(
                    "beta1_power", dtype=tf.float32,
                    initializer=self._beta1, trainable=False)
                self._beta2_power = tf.get_variable(
                    "beta2_power", dtype=tf.float32,
                    initializer=self._beta2, trainable=False)

        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "vhat", self._name)

    # ------------------------------------------------------------------
    # Prepare tensors
    # ------------------------------------------------------------------
    def _prepare(self):
        self._lr_t = tf.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = tf.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = tf.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = tf.convert_to_tensor(self._epsilon, name="epsilon")

    # ------------------------------------------------------------------
    # Dense update
    # ------------------------------------------------------------------
    def _apply_dense(self, grad, var):
        beta1_power = tf.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = tf.cast(self._beta2_power, var.dtype.base_dtype)
        lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = tf.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = tf.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = tf.cast(self._epsilon_t, var.dtype.base_dtype)

        lr = lr_t * tf.sqrt(1 - beta2_power) / (1 - beta1_power)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        vhat = self.get_slot(var, "vhat")

        m_t = tf.assign(m, beta1_t * m + (1 - beta1_t) * grad,
                        use_locking=self._use_locking)
        v_t = tf.assign(v, beta2_t * v + (1 - beta2_t) * tf.square(grad),
                        use_locking=self._use_locking)

        vhat_t = tf.assign(vhat, tf.maximum(vhat, v_t))
        v_sqrt = tf.sqrt(vhat_t)
        var_update = tf.assign_sub(var, lr * m_t / (v_sqrt + epsilon_t),
                                   use_locking=self._use_locking)
        return tf.group(var_update, m_t, v_t, vhat_t)

    # ------------------------------------------------------------------
    # Sparse/shared updates (optional)
    # ------------------------------------------------------------------
    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        beta1_power = tf.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = tf.cast(self._beta2_power, var.dtype.base_dtype)
        lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = tf.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = tf.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = tf.cast(self._epsilon_t, var.dtype.base_dtype)

        lr = lr_t * tf.sqrt(1 - beta2_power) / (1 - beta1_power)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        vhat = self.get_slot(var, "vhat")

        m_t = tf.assign(m, m * beta1_t, use_locking=self._use_locking)
        with tf.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, grad * (1 - beta1_t))

        v_t = tf.assign(v, v * beta2_t, use_locking=self._use_locking)
        with tf.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, tf.square(grad) * (1 - beta2_t))

        vhat_t = tf.assign(vhat, tf.maximum(vhat, v_t))
        v_sqrt = tf.sqrt(vhat_t)
        var_update = tf.assign_sub(var, lr * m_t / (v_sqrt + epsilon_t),
                                   use_locking=self._use_locking)
        return tf.group(var_update, m_t, v_t, vhat_t)

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: tf.scatter_add(x, i, v,
                                           use_locking=self._use_locking))

    # ------------------------------------------------------------------
    # Finish step (update power variables)
    # ------------------------------------------------------------------
    def _finish(self, update_ops, name_scope):
        with tf.control_dependencies(update_ops):
            update_beta1 = self._beta1_power.assign(
                self._beta1_power * self._beta1_t,
                use_locking=self._use_locking)
            update_beta2 = self._beta2_power.assign(
                self._beta2_power * self._beta2_t,
                use_locking=self._use_locking)
        return tf.group(*(update_ops + [update_beta1, update_beta2]),
                        name=name_scope)
