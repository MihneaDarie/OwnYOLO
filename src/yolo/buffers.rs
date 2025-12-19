use ndarray::{Array3, Array4 };

#[derive(Debug, Clone)]
pub struct C2FBuffer {
    pub initial: Array4<f32>,
    pub bottlenecks: Vec<BottleneckBuffer>,
    pub split_0: Array4<f32>,
    pub split_1: Array4<f32>,
    pub concat: Array4<f32>,
    pub last: Array4<f32>,
}

#[derive(Debug, Clone)]
pub struct BottleneckBuffer {
    pub cv1_out: Array4<f32>,
    pub cv2_out: Array4<f32>,
    pub add_out: Array4<f32>,
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
    pub anchor_outputs: Vec<Array3<f32>>,
    pub final_concat: Array3<f32>,
    pub bbox_pred: Array3<f32>,
    pub class_pred: Array3<f32>,
    pub bbox_coords: Array3<f32>,
    pub class_scores: Array3<f32>,
    pub final_output: Array3<f32>,
}

#[derive(Clone)]
pub struct DetectScaleBuffer {
    pub cv2_0_out: Array4<f32>,
    pub cv2_1_out: Array4<f32>,
    pub bbox_out: Array4<f32>,
    pub cv3_0_out: Array4<f32>,
    pub cv3_1_out: Array4<f32>,
    pub class_out: Array4<f32>,
    pub combined: Array4<f32>,
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
                conv_out: Array4::zeros((1, 16, 208, 208)),
            },
            model_1_buffer: ConvBuffer {
                conv_out: Array4::zeros((1, 32, 104, 104)),
            },
            model_2_buffer: C2FBuffer {
                initial: Array4::zeros((1, 32, 104, 104)),
                bottlenecks: vec![BottleneckBuffer {
                    cv1_out: Array4::zeros((1, 16, 104, 104)),
                    cv2_out: Array4::zeros((1, 16, 104, 104)),
                    add_out: Array4::zeros((1, 16, 104, 104)),
                }],
                split_0: Array4::zeros((1, 16, 104, 104)),
                split_1: Array4::zeros((1, 16, 104, 104)),
                concat: Array4::zeros((1, 48, 104, 104)),
                last: Array4::zeros((1, 32, 104, 104)),
            },
            model_3_buffer: ConvBuffer {
                conv_out: Array4::zeros((1, 64, 52, 52)),
            },
            model_4_buffer: C2FBuffer {
                initial: Array4::zeros((1, 64, 52, 52)),
                bottlenecks: vec![
                    BottleneckBuffer {
                        cv1_out: Array4::zeros((1, 32, 52, 52)),
                        cv2_out: Array4::zeros((1, 32, 52, 52)),
                        add_out: Array4::zeros((1, 32, 52, 52)),
                    },
                    BottleneckBuffer {
                        cv1_out: Array4::zeros((1, 32, 52, 52)),
                        cv2_out: Array4::zeros((1, 32, 52, 52)),
                        add_out: Array4::zeros((1, 32, 52, 52)),
                    },
                ],
                split_0: Array4::zeros((1, 32, 52, 52)),
                split_1: Array4::zeros((1, 32, 52, 52)),
                concat: Array4::zeros((1, 128, 52, 52)),
                last: Array4::zeros((1, 64, 52, 52)),
            },
            model_5_buffer: ConvBuffer {
                conv_out: Array4::zeros((1, 128, 26, 26)),
            },
            model_6_buffer: C2FBuffer {
                initial: Array4::zeros((1, 128, 26, 26)),
                bottlenecks: vec![
                    BottleneckBuffer {
                        cv1_out: Array4::zeros((1, 64, 26, 26)),
                        cv2_out: Array4::zeros((1, 64, 26, 26)),
                        add_out: Array4::zeros((1, 64, 26, 26)),
                    },
                    BottleneckBuffer {
                        cv1_out: Array4::zeros((1, 64, 26, 26)),
                        cv2_out: Array4::zeros((1, 64, 26, 26)),
                        add_out: Array4::zeros((1, 64, 26, 26)),
                    },
                ],
                split_0: Array4::zeros((1, 64, 26, 26)),
                split_1: Array4::zeros((1, 64, 26, 26)),
                concat: Array4::zeros((1, 256, 26, 26)),
                last: Array4::zeros((1, 128, 26, 26)),
            },
            model_7_buffer: ConvBuffer {
                conv_out: Array4::zeros((1, 256, 13, 13)),
            },
            model_8_buffer: C2FBuffer {
                initial: Array4::zeros((1, 256, 13, 13)),
                bottlenecks: vec![BottleneckBuffer {
                    cv1_out: Array4::zeros((1, 128, 13, 13)),
                    cv2_out: Array4::zeros((1, 128, 13, 13)),
                    add_out: Array4::zeros((1, 128, 13, 13)),
                }],
                split_0: Array4::zeros((1, 128, 13, 13)),
                split_1: Array4::zeros((1, 128, 13, 13)),
                concat: Array4::zeros((1, 384, 13, 13)),
                last: Array4::zeros((1, 256, 13, 13)),
            },
            model_9_buffer: SPPFBuffer {
                cv1_out: Array4::zeros((1, 128, 13, 13)),
                pool_1: Array4::zeros((1, 128, 13, 13)),
                pool_2: Array4::zeros((1, 128, 13, 13)),
                pool_3: Array4::zeros((1, 128, 13, 13)),
                concat: Array4::zeros((1, 512, 13, 13)),
                cv2_out: Array4::zeros((1, 256, 13, 13)),
            },
            
            up1_buffer: UpsampleBuffer {
                output: Array4::zeros((1, 256, 26, 26)),
            },
            concat1_buffer: ConcatBuffer {
                output: Array4::zeros((1, 384, 26, 26)),
            },
            
            model_12_buffer: C2FBuffer {
                initial: Array4::zeros((1, 128, 26, 26)),
                bottlenecks: vec![BottleneckBuffer {
                    cv1_out: Array4::zeros((1, 64, 26, 26)),
                    cv2_out: Array4::zeros((1, 64, 26, 26)),
                    add_out: Array4::zeros((1, 64, 26, 26)),
                }],
                split_0: Array4::zeros((1, 64, 26, 26)),
                split_1: Array4::zeros((1, 64, 26, 26)),
                concat: Array4::zeros((1, 192, 26, 26)),
                last: Array4::zeros((1, 128, 26, 26)),
            },
            
            up2_buffer: UpsampleBuffer {
                output: Array4::zeros((1, 128, 52, 52)),
            },
            concat2_buffer: ConcatBuffer {
                output: Array4::zeros((1, 192, 52, 52)),
            },
            
            model_15_buffer: C2FBuffer {
                initial: Array4::zeros((1, 64, 52, 52)),
                bottlenecks: vec![BottleneckBuffer {
                    cv1_out: Array4::zeros((1, 32, 52, 52)),
                    cv2_out: Array4::zeros((1, 32, 52, 52)),
                    add_out: Array4::zeros((1, 32, 52, 52)),
                }],
                split_0: Array4::zeros((1, 32, 52, 52)),
                split_1: Array4::zeros((1, 32, 52, 52)),
                concat: Array4::zeros((1, 96, 52, 52)),
                last: Array4::zeros((1, 64, 52, 52)),
            },
            
            model_16_buffer: ConvBuffer {
                conv_out: Array4::zeros((1, 64, 26, 26)),
            },
            
            concat3_buffer: ConcatBuffer {
                output: Array4::zeros((1, 192, 26, 26)),
            },
            
            model_18_buffer: C2FBuffer {
                initial: Array4::zeros((1, 128, 26, 26)),
                bottlenecks: vec![BottleneckBuffer {
                    cv1_out: Array4::zeros((1, 64, 26, 26)),
                    cv2_out: Array4::zeros((1, 64, 26, 26)),
                    add_out: Array4::zeros((1, 64, 26, 26)),
                }],
                split_0: Array4::zeros((1, 64, 26, 26)),
                split_1: Array4::zeros((1, 64, 26, 26)),
                concat: Array4::zeros((1, 192, 26, 26)),
                last: Array4::zeros((1, 128, 26, 26)),
            },
            
            model_19_buffer: ConvBuffer {
                conv_out: Array4::zeros((1, 128, 13, 13)),
            },
            
            concat4_buffer: ConcatBuffer {
                output: Array4::zeros((1, 384, 13, 13)),
            },
            
            model_21_buffer: C2FBuffer {
                initial: Array4::zeros((1, 256, 13, 13)),
                bottlenecks: vec![BottleneckBuffer {
                    cv1_out: Array4::zeros((1, 128, 13, 13)),
                    cv2_out: Array4::zeros((1, 128, 13, 13)),
                    add_out: Array4::zeros((1, 128, 13, 13)),
                }],
                split_0: Array4::zeros((1, 128, 13, 13)),
                split_1: Array4::zeros((1, 128, 13, 13)),
                concat: Array4::zeros((1, 384, 13, 13)),
                last: Array4::zeros((1, 256, 13, 13)),
            },
            
            model_22_buffer: DetectHeadBuffer {
                scale_outputs: [
                    DetectScaleBuffer {
                        cv2_0_out: Array4::zeros((1, 64, 52, 52)),
                        cv2_1_out: Array4::zeros((1, 64, 52, 52)),
                        bbox_out: Array4::zeros((1, 64, 52, 52)),
                        cv3_0_out: Array4::zeros((1, 80, 52, 52)),
                        cv3_1_out: Array4::zeros((1, 80, 52, 52)),
                        class_out: Array4::zeros((1, 80, 52, 52)),
                        combined: Array4::zeros((1, 144, 52, 52)),
                    },
                    DetectScaleBuffer {
                        cv2_0_out: Array4::zeros((1, 64, 26, 26)),
                        cv2_1_out: Array4::zeros((1, 64, 26, 26)),
                        bbox_out: Array4::zeros((1, 64, 26, 26)),
                        cv3_0_out: Array4::zeros((1, 80, 26, 26)),
                        cv3_1_out: Array4::zeros((1, 80, 26, 26)),
                        class_out: Array4::zeros((1, 80, 26, 26)),
                        combined: Array4::zeros((1, 144, 26, 26)),
                    },
                    DetectScaleBuffer {
                        cv2_0_out: Array4::zeros((1, 64, 13, 13)),
                        cv2_1_out: Array4::zeros((1, 64, 13, 13)),
                        bbox_out: Array4::zeros((1, 64, 13, 13)),
                        cv3_0_out: Array4::zeros((1, 80, 13, 13)),
                        cv3_1_out: Array4::zeros((1, 80, 13, 13)),
                        class_out: Array4::zeros((1, 80, 13, 13)),
                        combined: Array4::zeros((1, 144, 13, 13)),
                    },
                ],
                anchor_outputs: vec![
                    Array3::zeros((1, 144, 2704)),
                    Array3::zeros((1, 144, 676)),
                    Array3::zeros((1, 144, 169)),
                ],
                final_concat: Array3::zeros((1, 144, 3549)),
                bbox_pred: Array3::zeros((1, 64, 3549)),
                class_pred: Array3::zeros((1, 80, 3549)),
                bbox_coords: Array3::zeros((1, 4, 3549)),
                class_scores: Array3::zeros((1, 80, 3549)),
                final_output: Array3::zeros((1, 84, 3549)),
            },
        }
    }
}