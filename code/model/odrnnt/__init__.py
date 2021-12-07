from dataclasses import dataclass



@dataclass
class OnDeviceRNNTConfig:
    teacher_forcing_ratio: float = 1.0
    teacher_forcing_step: float = 0.01
    min_teacher_forcing_ratio: float = 0.9
    dropout: float = 0.3
    joint_ctc_attention: bool = False
    max_len: int = 400
    architecture: str = "rnnt"
    num_encoder_layers: int = 8
    num_decoder_layers: int = 2
    encoder_hidden_state_dim: int = 2048
    decoder_hidden_state_dim: int = 2048
    output_dim: int = 640
    rnn_type: str = "lstm"
    encoder_dropout_p: float = 0.2
    decoder_dropout_p: float = 0.2
    bidirectional: bool = False