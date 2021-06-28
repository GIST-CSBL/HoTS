import tensorflow as tf
from keras.losses import MSE, categorical_crossentropy, binary_crossentropy

EPSILON = 1e-5

class HoTSLoss(object):

    def __init__(self, negative_loss_weight=0.3, reg_loss_weight=10, sparsity_weight=0.001, conf_loss_weight=1.,
                 grid_size=25, attention_loss_weight=1, anchors=None, retina_loss_weight=2):
        self.conf_loss_weight = conf_loss_weight
        self.negative_loss_weight = negative_loss_weight
        self.reg_loss_weight = reg_loss_weight
        self.sparsity_weight = sparsity_weight
        self.grid_size = grid_size
        self.anchors = anchors
        self.attention_loss_weight = attention_loss_weight
        self.retina_weight = retina_loss_weight

    def tanh_to_sigmoid(self, value):
        return (value+1.)/2

    def compute_iou(self, gt_start, gt_end, pred_start, pred_end):
        with tf.name_scope('Intersections'):
            intersect_mins = tf.maximum(gt_start, pred_start)
            intersect_maxes = tf.minimum(gt_end, pred_end)
            intersect_hw = tf.maximum(intersect_maxes - intersect_mins, 0)
            intersections = intersect_hw#[..., 0]
        with tf.name_scope("Unions"):
            union_mins = tf.minimum(gt_start, pred_start)
            union_maxes = tf.maximum(gt_end, pred_end)
            union_hw = tf.maximum(union_maxes - union_mins, EPSILON)
            unions = union_hw#[..., 0]
        return tf.where(tf.equal(intersections, 0.0),
                     tf.zeros_like(intersections)+EPSILON,
                    tf.truediv(intersections, unions))

    def compute_hots_loss(self, y_true, y_pred):
        # Ground Truth
        gt_boxes = y_true[..., 0:2]
        gt_label = y_true[..., 2]
        # Prediction
        pred_boxes = y_pred[..., 0:2]
        pred_label = y_pred[..., 2]

        is_true = gt_label
        is_false = 1 - is_true
        n_true = tf.reduce_sum(is_true)
        n_false = tf.reduce_sum(is_false)

        gt_c = gt_boxes[..., 0]
        pred_c = pred_boxes[..., 0]
        c_loss = tf.abs(gt_c - pred_c) * self.reg_loss_weight * is_true

        gt_w = gt_boxes[..., 1]
        pred_w = pred_boxes[..., 1]
        w_loss = tf.abs(gt_w - pred_w) * self.reg_loss_weight * is_true

        grid_anchors = tf.cast(tf.reshape(self.anchors, [1, 1, len(self.anchors)]), tf.float32)
        '''
        gt_c = tf.ones_like(gt_boxes[..., 0])*self.grid_size + gt_boxes[..., 0]*self.grid_size
        gt_w = tf.exp(gt_boxes[..., 1])*grid_anchors
        pred_c = tf.ones_like(pred_boxes[..., 0])*self.grid_size + pred_boxes[..., 0]*self.grid_size
        pred_w = tf.exp(pred_boxes[..., 1])*grid_anchors
        gt_min = gt_c - gt_w/2
        gt_max = gt_c + gt_w/2
        pred_min = pred_c - pred_w/2
        pred_max = pred_c + pred_w/2
        iou_loss = (1 - self.compute_iou(gt_min, gt_max, pred_min, pred_max)) * self.reg_loss_weight * is_true
        c_loss = tf.square(gt_c - pred_c) * self.reg_loss_weight * is_true
        w_loss = tf.square(gt_w - pred_w) * self.reg_loss_weight * is_true
        '''

        # Retina
        pt = tf.maximum(pred_label, EPSILON)
        pt = tf.minimum(pt, 1 - EPSILON)
        retina_conf_obj = -((1-pt)**self.retina_weight)*tf.log(pt) * self.conf_loss_weight * is_true
        retina_conf_noobj = -((pt)**self.retina_weight)*tf.log(1-pt) * self.conf_loss_weight * self.negative_loss_weight * is_false

        obj_loss = retina_conf_obj + retina_conf_noobj

        total_loss = tf.reduce_sum(obj_loss)/(n_true+n_false) + tf.reduce_sum(c_loss+w_loss)/(n_true) # + tf.reduce_sum(iou_loss)/(len(self.anchors)*n_true)
        return total_loss