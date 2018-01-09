import tensorflow as tf
from base import BaseClass
import time

class Encoder(BaseClass):
    def __init__(self,d_model,d_k,d_v,sequence_length,h,batch_size,num_layer,Q,K_s,type='encoder',mask=None,dropout_keep_prob=None,use_residual_conn=True):
        super(Encoder, self).__init__(d_model,d_k,d_v,sequence_length,h,batch_size,num_layer=num_layer)
        self.Q=Q
        self.K_s=K_s
        self.type=type
        self.mask=mask
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.dropout_keep_prob=dropout_keep_prob
        self.use_residual_conn=use_residual_conn

    def encoder_fn(self):
        start = time.time()
        print("encoder_fn.started.")
        Q=self.Q
        K_s=self.K_s
        for layer_index in range(self.num_layer):
            Q, K_s=self.encoder_single_layer(Q,K_s,layer_index)
            print("encoder_fn.",layer_index,".Q:",Q,";K_s:",K_s)
        end = time.time()
        print("encoder_fn.ended.Q:",Q,";K_s:",K_s,";time spent:",(end-start))
        return Q,K_s



    def encoder_single_layer(self,Q,K_s,layer_index):
        """
        singel layer for encoder.each layers has two sub-layers:
        the first is multi-head self-attention mechanism; the second is position-wise fully connected feed-forward network.
        for each sublayer. use LayerNorm(x+Sublayer(x)). input and output of last dimension: d_model
        :param Q: shape should be:       [batch_size,sequence_length,d_model]
        :param K_s: shape should be:     [batch_size,sequence_length,d_model]
        :return:output: shape should be:[batch_size,sequence_length,d_model]
        """
        #1.1 the first is multi-head self-attention mechanism
        multi_head_attention_output=self.sub_layer_multi_head_attention(layer_index,
            Q,K_s,self.type,mask=self.mask,
            dropout_keep_prob=self.dropout_keep_prob) #[batch_size,sequence_length,d_model]
        print("after multi_head_attention, the shape is =================", multi_head_attention_output.get_shape())
    
        #1.2 use LayerNorm(x+Sublayer(x)). all dimension=512.
        multi_head_attention_output=self.sub_layer_layer_norm_residual_connection(K_s,
            multi_head_attention_output,layer_index,
            'encoder_multi_head_attention',dropout_keep_prob=self.dropout_keep_prob,
            use_residual_conn=self.use_residual_conn)
        print("after first layernorm, the shape is =================", multi_head_attention_output.get_shape())
        #2.1 the second is position-wise fully connected feed-forward network.
        postion_wise_feed_forward_output=self.sub_layer_postion_wise_feed_forward(multi_head_attention_output,
            layer_index,self.type)
        #2.2 use LayerNorm(x+Sublayer(x)). all dimension=512.
        postion_wise_feed_forward_output= self.sub_layer_layer_norm_residual_connection(multi_head_attention_output,
            postion_wise_feed_forward_output,layer_index,'encoder_postion_wise_ff',
            dropout_keep_prob=self.dropout_keep_prob)
        return  postion_wise_feed_forward_output,postion_wise_feed_forward_output