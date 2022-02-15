# train
batch_size = 8
epoch = 50000
n_warm_up_step = 4000
save_per_epoch = 10
lr = 1e-4

# Mode
max_sep_len = 2048
word_vec_dim = 256
glu_channel = 64

encoder_glu_layer = 1
decoder_n_layer = 6
decoder_head = 1
decoder_conv1d_filter_size = 256
decoder_output_size = 256
fft_conv1d_kernel = 3
fft_conv1d_padding = 1
duration_predictor_filter_size = 256
duration_predictor_kernel_size = 3
dropout = 0.1

input_channel = 66

input_f0_channel = 4

# 60(harm channel) + 4(ap channel)
output_channel = 64

energy_output_channel = 1
