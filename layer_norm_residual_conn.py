import tensorflow as tf
import time
"""
We employ a residual connection around each of the two sub-layers, followed by layer normalization.
That is, the output of each sub-layer is LayerNorm(x+ Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. """
class LayerNormResidualConnection(object):
    def __init__(self,x,y,layer_index,type,residual_dropout=0.1,use_residual_conn=True):
        self.x=x
        self.y=y
        self.layer_index=layer_index
        self.type=type
        self.residual_dropout=residual_dropout
        self.use_residual_conn=use_residual_conn

    #call residual connection and layer normalization
    def layer_norm_residual_connection(self):
        print("LayerNormResidualConnection.use_residual_conn:",self.use_residual_conn)
        ##if self.use_residual_conn:
        #    x_residual=self.residual_connection()
        #    x_layer_norm=self.layer_normalization(x_residual)
        #else:
        layer_norm = self.layer_normalization(self.y)
        return layer_norm

    def residual_connection(self):
        output=self.x + tf.nn.dropout(self.y, 1.0 - self.residual_dropout)
        return output

    # layer normalize the tensor x, averaging over the last dimension.
    def layer_normalization(self,x):
        """
        x should be:[batch_size,sequence_length,d_model]
        :return:
        """
        filter=x.get_shape()[-1] #last dimension of x. e.g. 512
        print("the dimension of x is ", x.get_shape())
        print("layer_normalization:==================>variable_scope:","layer_normalization"+str(self.layer_index)+self.type)
        with tf.variable_scope("layer_normalization"+str(self.layer_index)+self.type):
            # 1. normalize input by using  mean and variance according to last dimension
            mean=tf.reduce_mean(x,axis=-1,keep_dims=True) #[batch_size,sequence_length,1]
            variance=tf.reduce_mean(tf.square(x-mean),axis=-1,keep_dims=True) #[batch_size,sequence_length,1]
            norm_x=(x-mean)*tf.rsqrt(variance+1e-6) #[batch_size,sequence_length,d_model]
            # 2. re-scale normalized input back
            scale=tf.get_variable("layer_norm_scale",[filter],initializer=tf.ones_initializer) #[filter]
            bias=tf.get_variable("layer_norm_bias",[filter],initializer=tf.ones_initializer) #[filter]
            output=norm_x*scale+bias #[batch_size,sequence_length,d_model]
            return output #[batch_size,sequence_length,d_model]