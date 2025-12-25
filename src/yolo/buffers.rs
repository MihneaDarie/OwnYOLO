use ndarray::{Array3, Array4};

#[derive(Debug)]
pub struct C2FBuffer {
    pub initial: Array4<f32>,
    pub bottlenecks: Vec<BottleneckBuffer>,
    pub split_1: Array4<f32>,
    pub last: Array4<f32>,
}

#[derive(Debug)]
pub struct BottleneckBuffer {
    pub cv1_out: Array4<f32>,
    pub cv2_out: Array4<f32>,
}

pub struct ConvBuffer {
    pub conv_out: Array4<f32>,
}

pub struct SPPFBuffer {
    pub cv1_out: Array4<f32>,
    pub pool_1: Array4<f32>,
    pub pool_2: Array4<f32>,
    pub pool_3: Array4<f32>,
    pub concat: Array4<f32>,
    pub cv2_out: Array4<f32>,
}

pub struct DetectHeadBuffer {
    pub scale_outputs: [DetectScaleBuffer; 3],
    pub final_concat: Array3<f32>,
    pub class_pred: Array3<f32>,
    pub final_output: Array3<f32>,
}

pub struct DetectScaleBuffer {
    pub cv2_0_out: Array4<f32>,
    pub cv2_1_out: Array4<f32>,
    pub bbox_out: Array4<f32>,
    pub cv3_0_out: Array4<f32>,
    pub cv3_1_out: Array4<f32>,
    pub class_out: Array4<f32>,
}

pub struct UpsampleBuffer {
    pub output: Array4<f32>,
}

pub struct ConcatBuffer {
    pub output: Array4<f32>,
}

pub struct Buffers {
    pub model_0_buffer: ConvBuffer,
    pub model_1_buffer: ConvBuffer,
    pub model_2_buffer: C2FBuffer,
    pub model_3_buffer: ConvBuffer,
    pub model_4_buffer: C2FBuffer,
    pub model_5_buffer: ConvBuffer,
    pub model_6_buffer: C2FBuffer,
    pub model_7_buffer: ConvBuffer,
    pub model_8_buffer: C2FBuffer,
    pub model_9_buffer: SPPFBuffer,

    pub up1_buffer: UpsampleBuffer,
    pub concat1_buffer: ConcatBuffer,

    pub model_12_buffer: C2FBuffer,

    pub up2_buffer: UpsampleBuffer,
    pub concat2_buffer: ConcatBuffer,

    pub model_15_buffer: C2FBuffer,
    pub model_16_buffer: ConvBuffer,

    pub concat3_buffer: ConcatBuffer,

    pub model_18_buffer: C2FBuffer,
    pub model_19_buffer: ConvBuffer,

    pub concat4_buffer: ConcatBuffer,

    pub model_21_buffer: C2FBuffer,
    pub model_22_buffer: DetectHeadBuffer,
    
}

impl Buffers {
    pub fn new() -> Self {
        Self {
            model_0_buffer: ConvBuffer {
                conv_out: Array4::zeros((1, 16, 320, 320)),
            },
            model_1_buffer: ConvBuffer {
                conv_out: Array4::zeros((1, 32, 160, 160)),
            },
            model_2_buffer: C2FBuffer {
                initial: Array4::zeros((1, 32, 160, 160)),
                bottlenecks: vec![BottleneckBuffer {
                    cv1_out: Array4::zeros((1, 16, 160, 160)),
                    cv2_out: Array4::zeros((1, 16, 160, 160)),
                }],
                split_1: Array4::zeros((1, 16, 160, 160)),
                last: Array4::zeros((1, 32, 160, 160)),
            },
            model_3_buffer: ConvBuffer {
                conv_out: Array4::zeros((1, 64, 80, 80)),
            },
            model_4_buffer: C2FBuffer {
                initial: Array4::zeros((1, 64, 80, 80)),
                bottlenecks: vec![
                    BottleneckBuffer {
                        cv1_out: Array4::zeros((1, 32, 80, 80)),
                        cv2_out: Array4::zeros((1, 32, 80, 80)),
                    },
                    BottleneckBuffer {
                        cv1_out: Array4::zeros((1, 32, 80, 80)),
                        cv2_out: Array4::zeros((1, 32, 80, 80)),
                    },
                ],
                split_1: Array4::zeros((1, 32, 80, 80)),
                last: Array4::zeros((1, 64, 80, 80)),
            },
            model_5_buffer: ConvBuffer {
                conv_out: Array4::zeros((1, 128, 40, 40)),
            },
            model_6_buffer: C2FBuffer {
                initial: Array4::zeros((1, 128, 40, 40)),
                bottlenecks: vec![
                    BottleneckBuffer {
                        cv1_out: Array4::zeros((1, 64, 40, 40)),
                        cv2_out: Array4::zeros((1, 64, 40, 40)),
                    },
                    BottleneckBuffer {
                        cv1_out: Array4::zeros((1, 64, 40, 40)),
                        cv2_out: Array4::zeros((1, 64, 40, 40)),
                    },
                ],
                split_1: Array4::zeros((1, 64, 40, 40)),
                last: Array4::zeros((1, 128, 40, 40)),
            },
            model_7_buffer: ConvBuffer {
                conv_out: Array4::zeros((1, 256, 20, 20)),
            },
            model_8_buffer: C2FBuffer {
                initial: Array4::zeros((1, 256, 20, 20)),
                bottlenecks: vec![BottleneckBuffer {
                    cv1_out: Array4::zeros((1, 128, 20, 20)),
                    cv2_out: Array4::zeros((1, 128, 20, 20)),
                }],
                split_1: Array4::zeros((1, 128, 20, 20)),
                last: Array4::zeros((1, 256, 20, 20)),
            },
            model_9_buffer: SPPFBuffer {
                cv1_out: Array4::zeros((1, 128, 20, 20)),
                pool_1: Array4::zeros((1, 128, 20, 20)),
                pool_2: Array4::zeros((1, 128, 20, 20)),
                pool_3: Array4::zeros((1, 128, 20, 20)),
                concat: Array4::zeros((1, 512, 20, 20)),
                cv2_out: Array4::zeros((1, 256, 20, 20)),
            },

            up1_buffer: UpsampleBuffer {
                output: Array4::zeros((1, 256, 40, 40)),
            },
            concat1_buffer: ConcatBuffer {
                output: Array4::zeros((1, 384, 40, 40)),
            },

            model_12_buffer: C2FBuffer {
                initial: Array4::zeros((1, 128, 40, 40)),
                bottlenecks: vec![BottleneckBuffer {
                    cv1_out: Array4::zeros((1, 64, 40, 40)),
                    cv2_out: Array4::zeros((1, 64, 40, 40)),
                }],
                split_1: Array4::zeros((1, 64, 40, 40)),
                last: Array4::zeros((1, 128, 40, 40)),
            },

            up2_buffer: UpsampleBuffer {
                output: Array4::zeros((1, 128, 80, 80)),
            },
            concat2_buffer: ConcatBuffer {
                output: Array4::zeros((1, 192, 80, 80)),
            },

            model_15_buffer: C2FBuffer {
                initial: Array4::zeros((1, 64, 80, 80)),
                bottlenecks: vec![BottleneckBuffer {
                    cv1_out: Array4::zeros((1, 32, 80, 80)),
                    cv2_out: Array4::zeros((1, 32, 80, 80)),
                }],
                split_1: Array4::zeros((1, 32, 80, 80)),
                last: Array4::zeros((1, 64, 80, 80)),
            },

            model_16_buffer: ConvBuffer {
                conv_out: Array4::zeros((1, 64, 40, 40)),
            },

            concat3_buffer: ConcatBuffer {
                output: Array4::zeros((1, 192, 40, 40)),
            },

            model_18_buffer: C2FBuffer {
                initial: Array4::zeros((1, 128, 40, 40)),
                bottlenecks: vec![BottleneckBuffer {
                    cv1_out: Array4::zeros((1, 64, 40, 40)),
                    cv2_out: Array4::zeros((1, 64, 40, 40)),
                }],
                split_1: Array4::zeros((1, 64, 40, 40)),
                last: Array4::zeros((1, 128, 40, 40)),
            },

            model_19_buffer: ConvBuffer {
                conv_out: Array4::zeros((1, 128, 20, 20)),
            },

            concat4_buffer: ConcatBuffer {
                output: Array4::zeros((1, 384, 20, 20)),
            },

            model_21_buffer: C2FBuffer {
                initial: Array4::zeros((1, 256, 20, 20)),
                bottlenecks: vec![BottleneckBuffer {
                    cv1_out: Array4::zeros((1, 128, 20, 20)),
                    cv2_out: Array4::zeros((1, 128, 20, 20)),
                }],
                split_1: Array4::zeros((1, 128, 20, 20)),
                last: Array4::zeros((1, 256, 20, 20)),
            },

            model_22_buffer: DetectHeadBuffer {
                scale_outputs: [
                    DetectScaleBuffer {
                        cv2_0_out: Array4::zeros((1, 64, 80, 80)),
                        cv2_1_out: Array4::zeros((1, 64, 80, 80)),
                        bbox_out: Array4::zeros((1, 64, 80, 80)),
                        cv3_0_out: Array4::zeros((1, 80, 80, 80)),
                        cv3_1_out: Array4::zeros((1, 80, 80, 80)),
                        class_out: Array4::zeros((1, 80, 80, 80)),
                    },
                    DetectScaleBuffer {
                        cv2_0_out: Array4::zeros((1, 64, 40, 40)),
                        cv2_1_out: Array4::zeros((1, 64, 40, 40)),
                        bbox_out: Array4::zeros((1, 64, 40, 40)),
                        cv3_0_out: Array4::zeros((1, 80, 40, 40)),
                        cv3_1_out: Array4::zeros((1, 80, 40, 40)),
                        class_out: Array4::zeros((1, 80, 40, 40)),
                    },
                    DetectScaleBuffer {
                        cv2_0_out: Array4::zeros((1, 64, 20, 20)),
                        cv2_1_out: Array4::zeros((1, 64, 20, 20)),
                        bbox_out: Array4::zeros((1, 64, 20, 20)),
                        cv3_0_out: Array4::zeros((1, 80, 20, 20)),
                        cv3_1_out: Array4::zeros((1, 80, 20, 20)),
                        class_out: Array4::zeros((1, 80, 20, 20)),
                    },
                ],
                final_concat: Array3::zeros((1, 144, 8400)),
                class_pred: Array3::zeros((1, 80, 8400)),
                final_output: Array3::zeros((1, 84, 8400)),
            },
                    }
    }
}