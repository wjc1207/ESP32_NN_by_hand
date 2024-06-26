#include <stdio.h>
//define
#define CONV1D_1_C_IN 192
#define CONV1D_1_L 80
#define CONV1D_1_K 2
#define CONV1D_1_C_OUT 8
#define CONV1D_1_STRIDE 4

#define CONV1D_1_L_OUT ((CONV1D_1_L - CONV1D_1_K) / CONV1D_1_STRIDE + 1)
#define LINEAR_1_C_IN (CONV1D_1_C_OUT * CONV1D_1_L_OUT)

#define LINEAR_1_C_OUT 4

static float conv1d_1_weight[CONV1D_1_C_OUT][CONV1D_1_C_IN][CONV1D_1_K] = {{{0.0218, 0.0403}, {-0.0029, -0.0395}, {-0.0172, -0.0401}, {-0.0025, -0.0223}, {0.0135, -0.0416}, {0.0374, 0.0128}, {-0.0118, 0.021}, {0.0274, -0.0161}, {0.0002, 0.0366}, {0.0392, 0.0404}, {-0.045, 0.0316}, {0.0501, 0.028}, {0.0056, -0.0096}, {0.0218, 0.0341}, {0.0053, -0.052}, {-0.006, 0.0317}, {0.0109, 0.0463}, {-0.0388, 0.0376}, {-0.0099, 0.0495}, {-0.0298, -0.0096}, {0.0315, -0.0326}, {0.0373, -0.0227}, {-0.0295, 0.0231}, {-0.0363, 0.0447}, {-0.0324, 0.0411}, {0.0429, 0.0145}, {0.0163, 0.0066}, {-0.0382, 0.0134}, {0.0412, -0.0022}, {0.0075, -0.0291}, {-0.0157, -0.0393}, {0.0082, -0.0302}, {-0.0197, -0.04}, {-0.0218, -0.0361}, {-0.005, 0.0421}, {-0.0167, 0.0218}, {0.0472, -0.0318}, {-0.0539, 0.0144}, {0.0101, -0.0492}, {-0.0314, 0.0433}, {-0.0461, 0.0325}, {-0.038, -0.0052}, {-0.0089, -0.0214}, {-0.0214, 0.037}, {0.0433, 0.0275}, {0.0346, 0.0344}, {0.0406, -0.0109}, {-0.0272, -0.0321}, {-0.0364, 0.0366}, {0.0316, 0.0365}, {0.0293, 0.0361}, {0.0412, -0.0299}, {-0.0129, -0.0356}, {0.0045, 0.0214}, {0.0021, -0.0208}, {-0.0367, 0.0184}, {0.0337, 0.0196}, {0.0037, 0.0309}, {-0.0178, 0.0287}, {0.0378, -0.0388}, {-0.0257, -0.0444}, {-0.0121, -0.0256}, {-0.0508, 0.015}, {-0.0356, -0.0327}, {0.0262, 0.0019}, {-0.0219, -0.0501}, {-0.0175, -0.0014}, {0.0105, 0.0125}, {0.0285, 0.0102}, {0.0269, -0.0169}, {0.0342, -0.0012}, {-0.0127, 0.029}, {-0.0352, -0.0315}, {0.041, 0.0393}, {-0.0478, 0.0305}, {-0.0416, 0.0435}, {0.0248, 0.0029}, {-0.0173, 0.0243}, {0.0049, 0.035}, {0.0072, -0.0374}, {-0.0012, 0.0407}, {0.0316, -0.0343}, {0.0374, -0.0143}, {0.0381, -0.0334}, {0.0427, 0.0336}, {-0.0378, -0.0178}, {0.0392, -0.0157}, {-0.037, 0.0443}, {-0.0273, -0.0392}, {0.0216, 0.0139}, {-0.0398, -0.0038}, {-0.0321, -0.0314}, {0.0402, 0.013}, {-0.0394, 0.0256}, {-0.045, -0.0327}, {-0.0085, -0.0323}, {0.0558, -0.0447}, {0.0031, 0.0407}, {-0.0254, 0.0345}, {0.0193, 0.0222}, {-0.03, -0.0157}, {0.0293, -0.0493}, {-0.001, 0.054}, {-0.0033, 0.0228}, {-0.0053, 0.0097}, {0.0336, 0.0159}, {-0.0501, -0.0236}, {-0.0011, -0.0097}, {-0.0231, -0.0254}, {-0.0273, 0.0238}, {0.0429, -0.0204}, {-0.0401, 0.0483}, {0.0062, -0.0348}, {-0.0436, 0.0106}, {0.0116, 0.0301}, {-0.0324, -0.0113}, {-0.03, -0.0549}, {-0.0469, -0.01}, {-0.0516, -0.012}, {0.0242, 0.04}, {0.0046, -0.0373}, {0.0433, -0.0127}, {-0.0144, 0.0178}, {0.0114, -0.0439}, {-0.0257, 0.0334}, {0.029, -0.0461}, {-0.0348, 0.0255}, {-0.0326, 0.0095}, {-0.0284, -0.0317}, {-0.0382, 0.0404}, {0.0371, 0.0415}, {0.0391, -0.0364}, {0.0097, -0.0389}, {-0.0305, -0.0484}, {0.0003, 0.023}, {0.0203, 0.0122}, {-0.0107, 0.0089}, {0.0328, 0.0267}, {0.052, 0.0464}, {0.0072, 0.0342}, {0.0323, -0.0239}, {-0.0203, -0.0516}, {0.0018, 0.0021}, {-0.0193, 0.0129}, {0.0279, 0.0004}, {0.0053, -0.0262}, {0.0115, -0.0283}, {0.0513, -0.044}, {-0.048, 0.0013}, {-0.0366, 0.019}, {0.0448, -0.033}, {-0.0017, -0.0437}, {0.0108, -0.0264}, {0.0249, -0.032}, {0.0423, 0.0179}, {0.0466, -0.0168}, {-0.0328, 0.0079}, {-0.0322, -0.0022}, {-0.0319, -0.031}, {0.0117, -0.0258}, {0.039, -0.0286}, {0.0031, 0.0277}, {0.0535, -0.0344}, {0.0141, 0.0096}, {0.0119, 0.0082}, {-0.0128, -0.0362}, {0.0432, 0.0062}, {-0.0428, -0.0436}, {0.0432, 0.0364}, {0.033, 0.0287}, {0.0288, -0.0524}, {-0.0395, 0.0078}, {0.0357, -0.0484}, {0.0335, -0.0017}, {0.0172, -0.0228}, {-0.026, -0.0338}, {-0.0406, -0.04}, {-0.0251, 0.0117}, {0.0222, 0.0042}, {0.0391, -0.0048}, {-0.0373, 0.0517}, {-0.0254, -0.007}, {-0.0224, 0.047}, {-0.0086, 0.0284}, {-0.0091, -0.0402}, {0.0382, -0.0289}, {-0.0126, 0.0343}, {-0.0087, 0.0462}, {0.0142, -0.0134}, {-0.0379, 0.0377}, {0.0467, -0.0518}, {0.0419, -0.0073}}, {{0.0145, 0.0261}, {0.0523, 0.0187}, {0.0387, -0.0217}, {-0.0025, 0.0416}, {0.0414, 0.0191}, {0.0088, -0.0288}, {-0.0532, 0.0074}, {0.0362, -0.0011}, {-0.0128, 0.0316}, {-0.0564, -0.006}, {0.0404, 0.0242}, {-0.0008, -0.012}, {0.0404, -0.0211}, {-0.0062, -0.0374}, {0.0169, 0.0268}, {-0.0013, -0.0233}, {0.0035, -0.0032}, {0.0096, 0.0053}, {0.0117, -0.0386}, {0.0488, -0.0234}, {0.0125, -0.02}, {0.0167, 0.0069}, {-0.0297, 0.0254}, {-0.0402, 0.0466}, {-0.0429, 0.0467}, {-0.0048, 0.0155}, {0.0399, -0.0473}, {0.0017, 0.0084}, {0.0439, -0.0085}, {0.0207, -0.0063}, {-0.053, 0.0326}, {0.0445, -0.0372}, {-0.0315, -0.0507}, {0.0061, -0.0028}, {-0.023, 0.0397}, {0.0238, -0.0469}, {-0.0163, -0.0312}, {0.0395, -0.0314}, {0.0222, -0.0155}, {0.0116, -0.0521}, {-0.0241, 0.0124}, {-0.0217, -0.0218}, {-0.0369, 0.0411}, {-0.009, -0.0022}, {-0.0108, -0.0004}, {-0.0314, 0.0109}, {-0.0087, -0.0051}, {0.0471, -0.0272}, {0.0093, -0.0249}, {-0.0084, -0.0222}, {-0.0358, -0.0054}, {0.0191, -0.0109}, {0.0338, 0.0535}, {0.0302, -0.0391}, {0.0185, 0.0382}, {-0.0227, -0.0197}, {-0.0285, -0.0377}, {0.0214, 0.0145}, {-0.0037, 0.0351}, {0.0332, 0.0031}, {-0.03, -0.0326}, {0.0367, -0.0232}, {0.0172, 0.0127}, {0.0116, -0.0068}, {0.0586, 0.0207}, {0.015, 0.0447}, {0.0151, 0.0263}, {-0.0189, -0.0226}, {-0.0373, -0.004}, {0.0351, -0.0505}, {0.0551, 0.0013}, {0.0456, -0.0447}, {0.0213, 0.0201}, {0.0558, 0.016}, {-0.0034, 0.0372}, {-0.0093, 0.0028}, {-0.0264, 0.0483}, {0.0112, -0.0461}, {-0.007, 0.0427}, {0.0319, -0.0374}, {0.0224, -0.0234}, {-0.0122, -0.0532}, {-0.0395, -0.0269}, {-0.0315, -0.0124}, {-0.0478, 0.05}, {0.015, 0.0385}, {-0.0342, 0.0312}, {0.0154, 0.0422}, {-0.0311, -0.0402}, {-0.02, 0.0133}, {0.0281, 0.0353}, {-0.0125, 0.0391}, {-0.0183, 0.0141}, {-0.0459, -0.0136}, {-0.0198, -0.0383}, {-0.0225, 0.0513}, {0.0179, -0.0398}, {0.0268, -0.0325}, {0.0397, 0.0202}, {-0.0041, 0.0227}, {0.0369, 0.032}, {-0.0121, 0.0508}, {-0.0382, -0.0322}, {-0.0066, -0.0125}, {-0.0451, 0.0075}, {0.0279, -0.013}, {0.0375, -0.0068}, {-0.0152, 0.0415}, {-0.0417, 0.0134}, {0.009, -0.0333}, {-0.0017, -0.0347}, {0.018, -0.0349}, {-0.039, -0.0405}, {0.0018, -0.0047}, {-0.0311, 0.0435}, {0.0292, -0.0278}, {-0.0457, 0.0365}, {-0.03, -0.0209}, {0.022, -0.0332}, {-0.0337, -0.0306}, {-0.032, 0.03}, {0.0098, 0.0161}, {-0.02, 0.0172}, {0.0012, -0.0227}, {0.0492, -0.0424}, {-0.0197, -0.0287}, {0.0335, -0.0318}, {-0.0438, 0.0015}, {0.0407, -0.0124}, {-0.0177, -0.0551}, {-0.0311, -0.028}, {-0.009, -0.0012}, {0.0224, -0.0051}, {0.0112, -0.0292}, {0.0422, 0.0307}, {-0.0066, -0.009}, {0.0048, 0.0034}, {0.0136, 0.0177}, {-0.0164, 0.036}, {0.0353, -0.0323}, {0.0308, 0.0297}, {-0.0469, -0.001}, {-0.0361, 0.0448}, {-0.0501, -0.0426}, {-0.0084, 0.0444}, {0.0543, -0.0176}, {0.0389, 0.0011}, {0.016, -0.0255}, {-0.0241, -0.0239}, {-0.0494, -0.0309}, {0.0058, -0.0277}, {0.0259, -0.034}, {0.0213, 0.0124}, {0.0201, -0.019}, {0.0564, -0.0148}, {0.043, 0.0474}, {0.0156, 0.0441}, {0.0111, -0.0308}, {-0.0466, 0.0425}, {-0.011, 0.0081}, {0.0225, -0.0274}, {0.0257, -0.0075}, {-0.0143, -0.0057}, {-0.0464, 0.0266}, {0.0439, -0.0194}, {-0.007, -0.0005}, {-0.0093, 0.0172}, {0.0436, 0.0162}, {0.0038, -0.0493}, {0.0393, -0.0285}, {-0.0029, -0.0425}, {-0.03, 0.0006}, {-0.0095, 0.0313}, {-0.0047, -0.0459}, {0.0168, -0.022}, {0.0042, -0.0015}, {-0.0262, -0.0261}, {-0.0348, 0.0391}, {-0.0065, 0.0495}, {0.0301, 0.009}, {-0.0229, -0.0301}, {0.026, 0.0309}, {0.0281, -0.0157}, {0.0241, -0.0039}, {-0.0087, -0.0414}, {0.0138, -0.0288}, {0.02, 0.0153}, {-0.0195, 0.0373}, {-0.007, -0.0383}, {0.0373, -0.034}, {-0.0226, -0.013}, {-0.0482, -0.0437}}, {{0.0453, -0.0068}, {-0.0133, -0.0208}, {0.0124, -0.035}, {0.056, -0.0259}, {-0.0395, -0.0026}, {-0.0454, -0.025}, {0.0349, -0.0093}, {-0.0041, -0.0215}, {-0.0174, 0.0163}, {-0.0277, -0.0125}, {0.0558, -0.0183}, {-0.0053, -0.0358}, {-0.023, -0.0303}, {0.046, 0.0449}, {0.0092, 0.0404}, {0.0603, 0.0101}, {-0.0109, 0.0424}, {-0.026, -0.0387}, {0.0115, -0.0179}, {-0.0072, -0.0266}, {0.0227, 0.0049}, {0.0004, 0.0123}, {-0.0407, -0.0251}, {-0.0545, -0.0081}, {0.0401, -0.0241}, {-0.0424, 0.0026}, {0.0424, -0.0157}, {-0.0198, 0.0374}, {-0.0057, 0.0411}, {0.0186, 0.0373}, {0.0202, 0.0136}, {0.0196, -0.0307}, {-0.0015, -0.0007}, {0.0483, 0.0242}, {-0.01, 0.0552}, {0.0223, 0.0079}, {0.0144, -0.0414}, {-0.0039, -0.0182}, {0.0153, -0.0343}, {-0.0108, -0.0084}, {0.0298, -0.0277}, {0.0452, 0.0265}, {-0.0375, -0.0367}, {0.0167, -0.0006}, {0.0297, -0.0488}, {0.0199, 0.0406}, {-0.04, -0.0042}, {-0.0194, -0.0426}, {0.037, 0.0098}, {0.0072, 0.0273}, {0.0294, 0.0095}, {0.06, -0.0063}, {0.0448, 0.0047}, {0.0251, 0.0022}, {-0.0046, 0.0455}, {0.0571, 0.0192}, {-0.0379, 0.0432}, {0.0376, 0.0151}, {0.0114, 0.0247}, {0.0046, 0.014}, {-0.0374, -0.0304}, {0.0022, -0.0111}, {0.0036, -0.0012}, {-0.0357, -0.0268}, {0.0008, -0.0398}, {0.027, 0.0102}, {0.012, 0.0343}, {-0.0391, 0.0302}, {0.0147, 0.0293}, {0.0483, 0.0392}, {-0.0364, -0.0399}, {-0.0473, 0.0202}, {-0.0335, -0.0333}, {-0.0084, 0.0263}, {0.0307, 0.0197}, {0.0093, -0.01}, {0.0427, -0.0126}, {-0.0201, 0.0037}, {0.0105, 0.0035}, {-0.0271, 0.0264}, {-0.0218, -0.0282}, {0.0397, 0.0181}, {-0.0361, -0.028}, {-0.022, 0.0021}, {-0.0271, -0.0262}, {0.0103, 0.0068}, {-0.0069, 0.057}, {0.0175, 0.0268}, {-0.0464, 0.0153}, {0.0193, -0.0172}, {0.0144, 0.0026}, {-0.0314, -0.022}, {-0.0347, 0.0004}, {0.0516, -0.0178}, {-0.0031, 0.0306}, {-0.0131, -0.0059}, {0.0527, 0.0402}, {0.0391, -0.0349}, {0.0256, 0.0467}, {0.0485, -0.0372}, {0.0108, -0.0193}, {-0.0404, 0.0226}, {-0.035, -0.041}, {0.0117, 0.0194}, {0.0256, -0.017}, {0.0546, 0.0308}, {0.0427, -0.0301}, {-0.0279, 0.0152}, {-0.0404, 0.0149}, {0.0017, -0.0495}, {0.0054, 0.0485}, {-0.0196, -0.033}, {-0.0334, 0.0194}, {-0.0349, 0.0132}, {0.0122, -0.0357}, {-0.0234, 0.0463}, {0.0016, 0.0144}, {-0.0206, 0.0382}, {0.0135, -0.0217}, {-0.0019, -0.0047}, {0.0374, 0.0348}, {0.0328, 0.0344}, {-0.0242, -0.0423}, {0.0045, 0.0222}, {-0.012, 0.037}, {-0.0079, 0.0052}, {-0.0245, 0.0183}, {-0.0236, -0.0196}, {0.009, 0.0422}, {0.0193, -0.0167}, {0.003, -0.0105}, {-0.0523, -0.0189}, {-0.0063, -0.0218}, {-0.0416, -0.0457}, {-0.0323, -0.0358}, {0.0388, 0.015}, {-0.0346, 0.0459}, {-0.0513, 0.0212}, {0.0083, -0.0347}, {-0.0196, -0.0368}, {-0.0323, -0.0084}, {-0.0336, 0.0129}, {0.0216, 0.0038}, {0.0262, -0.0166}, {-0.0069, 0.0465}, {0.0079, 0.025}, {-0.0139, -0.0162}, {-0.02, 0.0153}, {-0.04, -0.036}, {-0.0424, 0.0089}, {-0.0272, -0.0419}, {-0.0298, -0.0136}, {-0.0145, -0.0406}, {0.0606, 0.0258}, {0.0297, 0.0465}, {0.0121, -0.0069}, {-0.0191, 0.019}, {-0.0468, 0.0223}, {0.0077, 0.0405}, {-0.018, -0.0146}, {0.048, -0.0414}, {-0.0524, -0.0113}, {-0.05, 0.0213}, {0.0087, -0.0194}, {0.0066, -0.0072}, {-0.0056, 0.0478}, {0.0557, -0.0026}, {-0.03, -0.0262}, {0.0192, 0.0196}, {-0.0018, -0.0505}, {0.0118, 0.0554}, {0.0276, -0.0358}, {-0.0007, -0.0389}, {-0.028, 0.0074}, {-0.055, -0.0222}, {-0.0361, -0.02}, {-0.0296, -0.0127}, {0.0565, 0.0371}, {-0.0363, -0.0086}, {0.0245, -0.0568}, {-0.0218, -0.047}, {-0.0113, 0.004}, {0.0401, -0.0159}, {0.0039, 0.0138}, {-0.036, -0.0411}, {0.0254, -0.0074}, {-0.0474, 0.031}, {-0.0161, 0.0067}, {-0.02, 0.0298}, {0.0465, -0.0032}, {0.0034, 0.0038}, {0.0259, 0.0227}}, {{0.0284, 0.0496}, {0.0247, 0.0271}, {-0.02, -0.0017}, {0.0351, 0.0065}, {0.0429, -0.0352}, {0.0217, -0.0097}, {-0.0371, -0.0533}, {-0.0121, -0.0468}, {-0.0417, 0.0227}, {0.0242, 0.027}, {0.0208, 0.0291}, {0.0253, -0.0232}, {0.0226, 0.0331}, {-0.0496, 0.0138}, {-0.0276, 0.007}, {0.0442, -0.0109}, {-0.0347, 0.0105}, {0.0424, 0.0063}, {-0.0203, -0.0215}, {0.0414, -0.0074}, {-0.0441, -0.0388}, {-0.0432, -0.0188}, {0.0244, 0.0362}, {-0.0054, 0.0367}, {0.0198, 0.0306}, {-0.0368, 0.0128}, {0.0129, -0.0285}, {0.0372, 0.031}, {0.0342, -0.0175}, {-0.045, 0.0346}, {0.0095, 0.0203}, {0.0492, -0.0189}, {0.0043, -0.044}, {-0.0388, 0.0501}, {0.0298, 0.0493}, {0.0244, 0.0412}, {-0.0385, 0.0474}, {-0.0109, -0.0136}, {-0.0252, 0.0226}, {-0.0477, -0.0421}, {0.0237, 0.041}, {-0.0255, -0.0225}, {0.0021, 0.007}, {0.032, -0.026}, {-0.0259, 0.0193}, {0.01, 0.0388}, {-0.0035, -0.0362}, {0.0257, -0.0367}, {0.04, 0.0253}, {-0.0207, -0.0522}, {0.0179, 0.0311}, {0.0314, 0.0144}, {-0.0418, -0.0068}, {-0.0104, 0.0105}, {-0.048, -0.0196}, {-0.0318, 0.0052}, {-0.0483, 0.0362}, {0.0132, 0.0622}, {0.0195, -0.0288}, {0.026, -0.0337}, {0.0213, 0.0054}, {0.024, 0.0109}, {-0.0104, -0.0176}, {-0.0201, -0.0468}, {-0.0128, -0.0072}, {0.054, 0.0486}, {0.0347, -0.0439}, {0.0372, 0.0292}, {-0.006, -0.0373}, {0.0447, 0.0321}, {-0.0039, -0.0257}, {-0.0239, 0.0158}, {0.0046, 0.0088}, {-0.0262, -0.0486}, {0.0089, -0.0193}, {0.0239, -0.0199}, {0.0433, -0.0058}, {-0.008, 0.0019}, {0.0199, -0.0237}, {-0.051, -0.0024}, {-0.0377, 0.0315}, {0.0496, -0.0303}, {0.0499, 0.0166}, {0.0255, 0.0172}, {-0.0004, -0.0378}, {0.0049, 0.0141}, {-0.0239, -0.0255}, {0.0056, -0.0447}, {0.0143, -0.0128}, {-0.0114, 0.0173}, {0.0182, 0.011}, {0.0355, 0.0176}, {-0.0151, -0.0296}, {0.0552, 0.0103}, {0.0222, -0.0477}, {-0.0283, -0.0047}, {-0.0018, -0.0038}, {0.008, -0.0052}, {-0.0338, 0.039}, {0.0401, -0.0169}, {0.031, 0.0254}, {-0.0355, 0.0047}, {-0.0337, 0.0261}, {-0.0338, -0.0219}, {0.036, -0.0395}, {-0.0177, -0.0083}, {-0.0276, 0.0091}, {0.0334, 0.0119}, {-0.0456, 0.0164}, {0.0185, 0.0375}, {-0.0084, 0.0022}, {-0.0188, -0.0149}, {-0.0363, 0.0311}, {0.0457, 0.0402}, {-0.0527, -0.0021}, {0.0463, 0.0099}, {0.0044, 0.0214}, {0.0426, -0.005}, {-0.0442, -0.04}, {0.0393, -0.0298}, {-0.0499, -0.0145}, {0.0119, -0.0109}, {0.0116, 0.0058}, {-0.0225, -0.0334}, {0.041, -0.01}, {0.0102, -0.0286}, {0.0217, 0.0305}, {-0.0298, -0.0051}, {-0.0242, 0.0026}, {0.0126, 0.0358}, {0.0417, -0.0247}, {0.0178, -0.0099}, {0.0358, -0.0163}, {-0.0163, -0.0505}, {-0.0231, 0.0371}, {-0.002, 0.0408}, {0.0291, 0.0522}, {-0.0055, 0.0219}, {0.0021, -0.0437}, {-0.0394, -0.0075}, {0.0157, 0.0282}, {0.0342, 0.0596}, {-0.0295, 0.0397}, {0.0098, 0.0366}, {0.023, -0.0328}, {-0.0217, -0.04}, {-0.0285, -0.0128}, {0.0221, -0.0253}, {-0.0168, -0.0092}, {0.0325, 0.0214}, {-0.0083, 0.0354}, {-0.0227, 0.0005}, {0.0274, -0.0016}, {0.0204, 0.0423}, {-0.0219, -0.051}, {0.0354, -0.0067}, {-0.0332, 0.0268}, {-0.0265, -0.0389}, {0.0581, -0.0301}, {-0.0004, -0.024}, {0.0008, -0.0224}, {-0.0348, 0.0028}, {-0.0425, -0.0215}, {-0.0492, 0.0001}, {0.0429, 0.0093}, {-0.016, 0.061}, {-0.0269, -0.0485}, {-0.0415, 0.0449}, {-0.0308, 0.0249}, {-0.0452, 0.0267}, {0.0065, -0.0167}, {0.0222, 0.0154}, {-0.0163, -0.0083}, {0.0265, 0.0097}, {-0.0114, 0.0349}, {0.0121, -0.0152}, {-0.0487, 0.0038}, {-0.0297, 0.0085}, {-0.0207, 0.0015}, {0.0464, -0.0008}, {0.0464, 0.0106}, {-0.0491, 0.019}, {-0.0144, -0.0175}, {0.0157, -0.0089}, {0.0282, -0.0027}, {0.0458, 0.0225}, {-0.0241, -0.0258}, {0.0223, 0.0018}, {-0.0276, -0.0082}, {0.0003, -0.0046}, {0.0118, -0.0499}, {-0.0168, -0.0224}}, {{-0.0419, 0.0386}, {-0.0186, -0.0015}, {-0.0428, 0.0158}, {-0.0095, 0.03}, {0.0052, 0.0447}, {-0.0044, 0.0114}, {0.0167, -0.001}, {-0.0375, -0.019}, {-0.0316, -0.019}, {0.0282, -0.0151}, {0.0157, 0.0326}, {0.0148, 0.0237}, {-0.0268, 0.0323}, {-0.0246, -0.0351}, {-0.0459, 0.017}, {0.0429, -0.0282}, {0.0163, -0.0045}, {0.0181, 0.0424}, {-0.0008, -0.0129}, {-0.0351, -0.0163}, {0.0218, -0.0575}, {-0.0342, 0.0068}, {-0.0231, 0.02}, {0.0455, 0.0196}, {-0.0233, 0.0114}, {-0.0093, -0.0091}, {-0.0345, -0.0419}, {-0.0481, -0.025}, {-0.0133, 0.0537}, {0.0138, 0.0374}, {0.0487, -0.0063}, {-0.0262, -0.0047}, {-0.0396, 0.0054}, {0.017, 0.0343}, {0.0126, -0.0144}, {0.0171, -0.0284}, {0.0052, -0.0118}, {-0.0087, 0.0156}, {0.0207, -0.0543}, {-0.0187, -0.0178}, {0.0099, 0.003}, {0.0161, -0.0409}, {0.0072, 0.0416}, {-0.0156, 0.0367}, {0.0346, -0.0595}, {-0.0506, 0.0361}, {0.0217, 0.0245}, {0.0294, 0.0281}, {0.0285, 0.0136}, {0.02, -0.0309}, {0.0261, 0.0109}, {0.0398, 0.0435}, {-0.0327, 0.0212}, {-0.0454, 0.0224}, {0.0513, 0.0537}, {-0.0285, -0.0055}, {-0.0135, -0.0518}, {0.0247, -0.0003}, {-0.0214, -0.0227}, {-0.0104, 0.0596}, {0.0097, -0.0046}, {-0.0132, -0.0339}, {-0.0425, -0.0555}, {0.002, -0.0217}, {-0.0172, 0.0128}, {0.024, 0.0389}, {0.0511, 0.0404}, {0.0138, -0.0048}, {0.0204, 0.0312}, {-0.0545, -0.0339}, {-0.0072, 0.0481}, {-0.0113, 0.0169}, {0.0384, 0.0037}, {-0.0102, 0.0224}, {0.0149, -0.0469}, {-0.0334, -0.0472}, {-0.0204, 0.0202}, {0.0321, -0.0132}, {0.0035, 0.0375}, {0.042, -0.0436}, {-0.0396, -0.0535}, {-0.0077, 0.0517}, {0.0313, -0.0181}, {0.0161, 0.0367}, {-0.0301, 0.0193}, {-0.0002, 0.0237}, {0.0312, -0.0484}, {0.0185, -0.0255}, {0.0144, 0.0531}, {-0.0016, -0.0448}, {0.0422, -0.0209}, {0.0089, -0.0413}, {-0.059, -0.029}, {0.0085, -0.028}, {-0.0172, 0.003}, {-0.0083, -0.0069}, {0.0336, -0.0428}, {-0.0263, -0.0241}, {-0.0329, -0.0555}, {-0.0079, 0.0264}, {-0.0161, -0.0108}, {0.0432, 0.0125}, {-0.0187, -0.0128}, {-0.0104, -0.0204}, {-0.037, -0.0499}, {0.0257, -0.0107}, {-0.0461, 0.045}, {0.0089, -0.0031}, {-0.0238, 0.0344}, {-0.0032, 0.0168}, {0.0458, 0.0334}, {-0.0505, 0.0275}, {0.0435, -0.026}, {-0.0218, -0.0239}, {0.0408, 0.0474}, {0.0085, -0.0142}, {-0.0201, -0.0214}, {0.0196, -0.011}, {0.0048, 0.0335}, {0.0087, 0.0317}, {0.01, 0.0419}, {0.0242, -0.0107}, {-0.0265, -0.0199}, {-0.0439, -0.0232}, {0.0225, 0.036}, {-0.0297, -0.0444}, {0.0036, -0.0266}, {0.0144, 0.0348}, {-0.0003, -0.0444}, {-0.0006, -0.0037}, {-0.0418, -0.027}, {0.0648, 0.0067}, {0.017, 0.0345}, {-0.0165, -0.0269}, {-0.0211, -0.0308}, {-0.0375, 0.0271}, {0.0659, 0.0231}, {0.0421, 0.0534}, {-0.0252, 0.045}, {-0.0301, 0.0237}, {0.0209, -0.0024}, {-0.038, -0.0346}, {0.0146, -0.0277}, {-0.0217, 0.0643}, {0.0305, 0.0346}, {-0.0315, 0.0041}, {0.0426, 0.002}, {0.0378, 0.0137}, {0.0311, -0.0312}, {0.0365, -0.0457}, {0.0425, 0.0215}, {-0.0113, -0.0569}, {-0.0041, -0.0239}, {0.0265, -0.0034}, {0.0145, 0.0363}, {0.0411, 0.015}, {-0.0507, -0.0514}, {0.0, 0.0247}, {0.0451, -0.0131}, {-0.0442, -0.0301}, {-0.0178, -0.0073}, {0.001, 0.0041}, {0.0007, -0.0118}, {-0.0393, 0.0003}, {-0.026, -0.0417}, {-0.0115, -0.0027}, {-0.0308, -0.0339}, {0.0238, -0.0098}, {0.0244, -0.0163}, {0.0067, -0.0464}, {0.0303, 0.0314}, {-0.0048, -0.0443}, {0.0532, 0.0513}, {-0.019, -0.0468}, {0.0207, -0.0379}, {0.0257, -0.0462}, {-0.0348, 0.0051}, {0.0113, -0.0127}, {-0.0236, -0.0083}, {0.0047, 0.0111}, {-0.0401, 0.0209}, {0.0288, -0.0083}, {0.004, -0.0209}, {0.0248, 0.0324}, {-0.0012, 0.0011}, {-0.0361, 0.0406}, {0.0381, -0.0128}, {0.0128, 0.0275}, {0.0136, -0.0097}, {0.0126, -0.0445}, {0.009, 0.0048}, {-0.035, 0.0253}}, {{-0.0122, -0.046}, {-0.0054, -0.0122}, {0.0266, 0.019}, {0.0594, 0.0383}, {0.0207, 0.052}, {0.0226, 0.0217}, {0.0167, 0.0279}, {-0.0014, 0.0445}, {0.0433, 0.0211}, {0.03, -0.0044}, {0.0337, -0.0027}, {0.0064, -0.015}, {-0.0409, 0.0064}, {0.028, -0.0454}, {0.0295, -0.0415}, {-0.035, -0.0287}, {-0.0274, -0.0434}, {-0.0317, 0.001}, {0.0075, 0.0459}, {-0.0131, 0.0178}, {-0.0456, -0.0257}, {0.0237, -0.02}, {-0.0304, -0.0376}, {-0.0458, -0.0246}, {-0.0232, -0.0021}, {0.0171, -0.0118}, {0.0386, -0.0187}, {0.013, -0.039}, {-0.0294, -0.0306}, {-0.0384, 0.0022}, {-0.03, 0.0321}, {-0.0559, -0.0444}, {-0.0067, 0.0442}, {0.0474, 0.0272}, {-0.0086, -0.039}, {-0.0357, -0.027}, {0.008, -0.0007}, {0.0403, -0.0494}, {-0.0532, -0.0228}, {-0.0265, 0.0159}, {0.024, 0.0441}, {0.0161, 0.0181}, {-0.0461, -0.0063}, {0.0172, 0.042}, {-0.0079, -0.0116}, {-0.0378, 0.0377}, {0.0238, -0.018}, {0.0203, -0.0304}, {0.0456, 0.0484}, {0.0317, 0.0326}, {0.0138, 0.0168}, {-0.0171, 0.0244}, {0.0154, -0.0287}, {0.0269, -0.002}, {0.0196, 0.0133}, {-0.0467, -0.0228}, {0.0027, -0.0136}, {0.0451, 0.0369}, {-0.0255, 0.0091}, {0.0154, -0.0359}, {0.008, 0.0395}, {-0.0036, 0.0208}, {-0.0151, -0.0266}, {0.0212, -0.015}, {-0.0309, 0.0344}, {-0.0223, -0.0197}, {-0.0337, 0.0288}, {0.0272, -0.0448}, {-0.0258, -0.0368}, {-0.0454, 0.0538}, {-0.0211, -0.0365}, {0.0021, -0.0127}, {0.0135, 0.003}, {0.0179, -0.0115}, {0.0301, -0.0113}, {-0.0162, -0.0202}, {0.0336, 0.0377}, {-0.0342, 0.0208}, {0.002, -0.0469}, {0.016, -0.0518}, {-0.0417, -0.0129}, {0.0323, -0.0378}, {0.0395, -0.0358}, {-0.0183, 0.0421}, {0.01, 0.0462}, {-0.0121, 0.0483}, {0.044, 0.0295}, {0.0018, -0.0058}, {-0.0214, 0.0033}, {-0.0414, -0.0468}, {0.0368, -0.0196}, {-0.0473, -0.0371}, {-0.0033, -0.0267}, {0.0438, -0.0008}, {-0.0236, -0.004}, {0.026, 0.0013}, {0.04, 0.0091}, {-0.022, -0.0474}, {0.0215, -0.0212}, {0.0315, 0.0268}, {-0.0054, 0.0094}, {0.0516, 0.0331}, {0.0106, -0.0088}, {0.0158, 0.0172}, {0.0339, 0.0309}, {-0.035, 0.0317}, {-0.0013, -0.0165}, {-0.0259, -0.0403}, {-0.0017, -0.0123}, {-0.027, -0.0413}, {-0.0289, -0.026}, {-0.0256, 0.0362}, {0.0425, 0.0375}, {0.0187, -0.005}, {-0.0257, 0.0088}, {-0.0016, -0.0068}, {-0.0453, -0.0117}, {-0.0389, -0.0381}, {0.0548, -0.0287}, {0.0214, -0.0546}, {0.0443, -0.019}, {0.0143, 0.0016}, {-0.0154, -0.0394}, {0.0299, 0.0535}, {-0.0266, 0.0176}, {-0.0058, -0.008}, {-0.0214, -0.0318}, {-0.0256, -0.0203}, {-0.0044, 0.0087}, {0.016, -0.039}, {0.0075, 0.0362}, {-0.0122, -0.0285}, {0.0247, -0.0274}, {0.0291, -0.0429}, {-0.004, -0.0394}, {0.0576, 0.0161}, {0.0477, 0.0445}, {-0.0173, -0.0328}, {-0.0306, -0.0138}, {-0.0355, 0.0129}, {0.0546, -0.0217}, {-0.0366, -0.0298}, {-0.0024, -0.0191}, {-0.0241, -0.0012}, {0.015, -0.0116}, {0.0476, 0.047}, {0.0248, 0.0519}, {-0.0182, 0.0364}, {0.0421, 0.0453}, {0.0115, 0.0309}, {0.0176, -0.0116}, {-0.0463, 0.0071}, {0.0303, -0.0212}, {-0.0045, 0.0017}, {-0.0153, 0.0491}, {-0.0083, -0.0463}, {0.002, -0.048}, {-0.0411, 0.0234}, {-0.0029, 0.0301}, {-0.0019, -0.0052}, {0.0564, 0.0466}, {0.0155, 0.0414}, {-0.0473, -0.0545}, {0.0122, 0.0262}, {-0.0099, -0.022}, {-0.0368, 0.0381}, {0.0008, -0.0241}, {-0.0504, 0.0255}, {-0.0321, -0.0321}, {-0.0359, 0.0281}, {0.0314, 0.048}, {-0.035, -0.0099}, {0.0357, -0.0383}, {0.0377, 0.0425}, {0.0325, -0.0117}, {-0.0131, 0.0191}, {-0.0047, 0.0346}, {-0.0149, -0.0126}, {0.0453, 0.0222}, {-0.0124, 0.0183}, {0.0494, -0.031}, {-0.0409, 0.0077}, {-0.0062, -0.0248}, {0.0128, -0.0047}, {-0.0033, 0.045}, {-0.0417, -0.0268}, {0.0064, 0.0241}, {-0.0076, -0.0034}, {0.032, 0.0402}, {0.0128, 0.0299}, {0.0137, -0.0064}, {0.027, -0.0023}}, {{-0.0406, 0.0415}, {0.0076, -0.0477}, {-0.0234, 0.0148}, {0.0048, -0.0173}, {-0.0068, 0.0336}, {-0.0243, 0.0123}, {-0.0463, 0.0239}, {-0.0394, -0.0245}, {-0.005, 0.0071}, {0.0121, 0.0463}, {0.0327, -0.029}, {0.0158, -0.0248}, {-0.0306, -0.0311}, {0.0393, -0.0371}, {-0.0336, -0.0051}, {0.0154, -0.002}, {-0.0017, 0.039}, {0.0136, -0.0096}, {-0.0125, 0.0347}, {0.0198, 0.0246}, {0.0044, 0.0211}, {-0.0283, 0.0275}, {0.0114, -0.0441}, {0.0556, -0.0276}, {0.002, 0.0388}, {0.0329, -0.0093}, {-0.005, -0.0474}, {-0.0425, -0.016}, {-0.035, -0.0055}, {0.0218, 0.021}, {0.0179, 0.0193}, {0.0459, -0.0383}, {0.0267, -0.0161}, {0.0283, 0.0023}, {0.0252, 0.0505}, {0.0407, -0.0243}, {0.0122, -0.0477}, {0.0119, 0.0039}, {-0.0135, 0.0164}, {0.039, 0.0473}, {0.0054, 0.035}, {-0.0196, 0.0477}, {-0.0514, 0.0446}, {-0.0081, -0.0372}, {0.0255, -0.0427}, {0.0272, 0.0066}, {0.0007, -0.0467}, {-0.0222, 0.0418}, {-0.0293, -0.0055}, {-0.0116, -0.0146}, {0.034, 0.0419}, {-0.0036, 0.0226}, {0.0231, 0.0057}, {0.0461, 0.0164}, {0.0303, -0.0279}, {0.0034, -0.0315}, {-0.0097, 0.0239}, {-0.0301, -0.0218}, {-0.0411, 0.0499}, {0.0317, 0.0079}, {-0.0246, -0.003}, {0.0168, 0.0378}, {0.0104, 0.0293}, {0.041, 0.0208}, {0.0075, -0.0255}, {-0.0439, -0.0036}, {-0.0268, -0.0196}, {-0.0147, 0.0247}, {-0.0245, -0.0458}, {-0.0173, 0.0522}, {0.038, -0.024}, {-0.0391, 0.0268}, {0.0346, -0.053}, {-0.0356, -0.0001}, {-0.0425, 0.0245}, {-0.0222, -0.0205}, {-0.0424, 0.0253}, {0.0074, -0.0264}, {-0.0437, -0.0297}, {0.0222, -0.0237}, {-0.0231, 0.0217}, {0.0101, 0.0081}, {-0.0453, -0.0059}, {0.0164, -0.0305}, {0.0526, 0.0089}, {-0.0567, 0.0168}, {0.0003, 0.0398}, {0.0157, 0.0098}, {0.0157, 0.012}, {-0.0129, 0.0445}, {-0.0171, 0.0246}, {-0.0016, 0.0055}, {0.0096, -0.0594}, {-0.0266, -0.0093}, {0.0192, -0.0446}, {0.0294, -0.0269}, {-0.0376, 0.0162}, {-0.0388, -0.0488}, {0.0361, 0.0497}, {0.045, 0.0198}, {-0.0034, 0.031}, {-0.0118, -0.0245}, {0.0401, 0.0298}, {-0.0447, -0.0231}, {0.0097, -0.0575}, {0.0499, -0.0306}, {0.0419, 0.024}, {-0.0117, 0.0522}, {0.0247, 0.0051}, {0.0149, -0.0425}, {-0.0384, 0.0209}, {-0.0183, -0.0336}, {0.0082, -0.0065}, {0.0088, 0.0223}, {-0.0163, -0.0246}, {0.0403, -0.0362}, {-0.0332, 0.0335}, {-0.0028, -0.0073}, {-0.0077, 0.0071}, {-0.0322, 0.0199}, {-0.0105, -0.0168}, {0.0249, 0.0005}, {0.0173, 0.0073}, {0.0522, 0.0529}, {-0.0153, -0.0156}, {0.0347, 0.0121}, {0.0247, -0.0228}, {0.0129, -0.0164}, {-0.0337, 0.0285}, {0.023, 0.0571}, {0.0259, 0.0359}, {-0.0053, -0.0301}, {0.0158, -0.0276}, {0.0055, 0.0229}, {0.0325, 0.0262}, {-0.0078, -0.0166}, {-0.0351, 0.0113}, {0.0463, -0.026}, {-0.0101, -0.0255}, {0.0279, -0.0017}, {0.0361, -0.0034}, {0.0052, 0.0562}, {-0.0324, -0.0302}, {-0.0437, -0.0278}, {0.0216, 0.0452}, {-0.0209, -0.0469}, {0.0382, 0.0282}, {-0.0172, 0.0526}, {0.039, 0.0189}, {0.0202, -0.0194}, {-0.0406, -0.026}, {-0.0108, 0.0309}, {-0.0064, -0.0095}, {-0.0357, 0.0265}, {-0.0499, 0.0412}, {-0.0022, -0.0412}, {-0.0, 0.0221}, {-0.0583, -0.0542}, {-0.0121, -0.017}, {0.0199, -0.0315}, {0.0094, -0.0013}, {-0.039, -0.0046}, {0.0421, -0.0494}, {-0.0458, 0.0116}, {-0.0096, 0.0382}, {0.0482, 0.0308}, {-0.0097, 0.0142}, {0.0515, 0.0518}, {0.0044, 0.0208}, {-0.0097, -0.0257}, {0.0077, -0.0267}, {0.0025, 0.0118}, {-0.0484, -0.0316}, {0.0351, -0.0006}, {0.0239, -0.049}, {-0.005, 0.0153}, {-0.0446, 0.0215}, {0.0328, -0.0326}, {0.0008, -0.0017}, {0.0023, 0.0183}, {-0.0291, -0.0086}, {0.0339, -0.0413}, {0.0382, 0.0414}, {-0.0291, 0.0034}, {-0.0479, -0.0351}, {0.0313, -0.0424}, {0.0329, -0.0276}, {-0.0157, 0.0081}, {-0.0392, 0.0397}, {0.0217, -0.0287}, {-0.0207, -0.0241}, {-0.0334, 0.0381}}, {{0.0313, -0.0477}, {-0.0136, -0.0205}, {0.0071, 0.0178}, {-0.0246, 0.0067}, {-0.046, 0.0296}, {0.0331, -0.0283}, {-0.0158, -0.0402}, {-0.0385, 0.0142}, {-0.0048, -0.0147}, {-0.0141, 0.0136}, {-0.0253, 0.0165}, {-0.0301, -0.012}, {0.0168, -0.0305}, {0.0326, 0.0078}, {-0.0273, 0.0376}, {0.0227, 0.0493}, {0.0024, -0.0143}, {0.0354, 0.028}, {-0.0163, 0.0164}, {0.0097, 0.0147}, {-0.0414, -0.0152}, {-0.0279, 0.0141}, {-0.0034, -0.0399}, {-0.0008, 0.0087}, {0.0324, 0.0396}, {-0.013, 0.0364}, {-0.0346, -0.0365}, {-0.0244, -0.0074}, {-0.0417, -0.0084}, {-0.0004, 0.0182}, {-0.0119, -0.0422}, {0.0332, -0.0014}, {-0.0371, 0.039}, {-0.0162, -0.0234}, {-0.0259, -0.036}, {0.0373, 0.0459}, {0.0181, -0.0328}, {0.0071, -0.0197}, {-0.0058, 0.053}, {-0.0095, -0.0366}, {-0.0221, 0.0375}, {0.0219, -0.0103}, {0.0276, 0.0288}, {-0.003, -0.0088}, {0.0379, 0.005}, {-0.0143, -0.0413}, {0.0055, 0.0539}, {-0.0165, 0.0328}, {0.051, -0.0288}, {0.0192, -0.0308}, {-0.0345, -0.0}, {-0.0064, -0.0063}, {0.0226, 0.0084}, {-0.0, 0.0034}, {0.0044, 0.0475}, {0.0322, 0.0141}, {0.0404, -0.0169}, {-0.0534, -0.0526}, {0.015, -0.02}, {0.0354, -0.0012}, {-0.0053, -0.019}, {-0.0347, 0.007}, {0.0455, -0.0401}, {0.0454, 0.0132}, {-0.0317, -0.0241}, {0.0217, 0.02}, {-0.0108, -0.0115}, {0.0144, 0.0435}, {-0.0374, 0.03}, {0.0223, 0.0388}, {-0.0447, -0.0399}, {0.0042, -0.0061}, {-0.0418, 0.0064}, {-0.0307, -0.0238}, {-0.0094, 0.001}, {-0.0483, -0.0234}, {0.0315, 0.0245}, {0.0124, -0.0276}, {0.029, 0.0021}, {-0.0157, 0.0006}, {-0.0129, -0.0461}, {-0.0151, -0.0488}, {0.0503, 0.0166}, {0.052, -0.0135}, {-0.0063, -0.0153}, {0.0153, -0.0242}, {0.0319, -0.0278}, {-0.0214, 0.0134}, {0.0029, 0.026}, {-0.005, -0.0317}, {0.0228, 0.0414}, {0.0522, -0.0253}, {0.0177, 0.0248}, {0.0179, 0.0071}, {0.0093, 0.0184}, {0.0087, -0.0173}, {-0.0158, -0.0493}, {0.0263, 0.0024}, {0.0001, -0.0418}, {-0.0163, 0.0135}, {-0.0288, 0.0507}, {0.0169, -0.0115}, {-0.023, 0.0091}, {-0.0314, 0.0166}, {0.0114, -0.0272}, {0.0176, -0.0196}, {-0.0333, -0.0101}, {-0.0258, 0.0303}, {-0.0364, -0.0224}, {0.0418, -0.0067}, {0.054, 0.0156}, {0.0328, -0.0021}, {0.0092, 0.0497}, {-0.0215, 0.0552}, {0.0321, -0.02}, {-0.0071, -0.0323}, {-0.0402, -0.0067}, {-0.0496, -0.0089}, {-0.0413, 0.0494}, {-0.03, 0.0182}, {-0.0117, -0.0303}, {0.0085, 0.0283}, {-0.0018, -0.0141}, {0.0218, -0.015}, {0.052, -0.0366}, {-0.0372, -0.0435}, {-0.0056, -0.001}, {-0.0338, -0.0284}, {-0.0529, -0.0154}, {-0.0248, -0.0353}, {0.0362, -0.0189}, {0.0308, 0.0278}, {0.0493, -0.0144}, {0.023, -0.0328}, {-0.0027, 0.0277}, {-0.0354, -0.0202}, {0.0128, 0.0321}, {0.0464, 0.045}, {-0.0441, -0.0138}, {-0.011, 0.0094}, {-0.0354, 0.0451}, {0.0326, -0.0046}, {0.0151, 0.0361}, {-0.0065, 0.0075}, {-0.0317, 0.0426}, {0.0078, 0.0562}, {0.0032, 0.0093}, {0.0105, -0.0189}, {0.0058, 0.041}, {-0.0064, 0.0291}, {0.0011, 0.0021}, {-0.0157, 0.0129}, {0.0367, 0.05}, {-0.0408, -0.0284}, {-0.0429, 0.0032}, {0.0096, -0.0065}, {-0.0365, 0.031}, {-0.0283, 0.0293}, {-0.0094, 0.0068}, {0.0324, -0.031}, {-0.0116, -0.0238}, {-0.0057, -0.0112}, {0.0193, 0.0112}, {-0.0367, -0.0244}, {-0.0118, 0.0421}, {-0.0404, -0.0463}, {0.0446, -0.0065}, {-0.0335, -0.0043}, {0.0001, 0.0143}, {-0.0009, -0.0082}, {-0.0194, -0.0186}, {-0.0082, -0.0113}, {-0.0207, -0.0131}, {-0.0483, 0.0539}, {-0.0313, -0.0153}, {0.0181, 0.0174}, {0.0309, 0.0348}, {0.0169, 0.0187}, {-0.0045, -0.0182}, {-0.0206, -0.0165}, {-0.0287, -0.0214}, {0.0153, 0.0436}, {0.0554, -0.0002}, {0.0144, 0.0077}, {0.0174, 0.0099}, {-0.0004, 0.0399}, {-0.0244, -0.0044}, {0.0453, 0.0115}, {-0.0144, -0.0504}, {0.0068, -0.0216}, {-0.0105, 0.0143}, {0.0576, 0.0315}}};
static float conv1d_1_bias[CONV1D_1_C_OUT] = {0.0435, -0.0465, -0.0459, 0.0278, 0.0478, -0.0028, -0.0259, 0.0242};

static float linear_1_weight[LINEAR_1_C_OUT][LINEAR_1_C_IN] = {{-0.0005, -0.0036, -0.0719, 0.0275, -0.0743, 0.0242, 0.0006, 0.0327, -0.0365, 0.0275, 0.0054, -0.0013, 0.0496, 0.067, 0.0142, -0.0076, -0.0113, 0.0719, -0.0256, 0.0274, -0.0623, 0.0005, 0.0007, -0.0165, 0.0092, 0.0647, 0.0388, -0.0143, 0.0174, 0.0547, -0.0512, -0.0117, -0.0379, -0.0439, 0.0297, 0.0699, 0.0331, -0.0262, -0.049, 0.0386, 0.0342, 0.0183, 0.0301, -0.0007, 0.0409, -0.053, -0.0711, -0.0681, 0.0653, -0.0486, -0.0261, -0.0648, -0.0024, -0.0306, -0.035, -0.0274, 0.0141, -0.0571, 0.0312, -0.0436, 0.0376, 0.0025, 0.0769, 0.0567, -0.0001, -0.0207, -0.0071, -0.0468, 0.0448, 0.053, 0.0222, -0.0368, -0.0074, -0.0151, 0.0434, -0.0167, -0.0304, -0.0298, -0.0142, -0.0933, 0.0166, -0.0361, 0.0454, 0.0136, -0.0053, 0.0614, 0.0072, 0.0427, 0.0356, 0.0444, 0.0021, 0.0444, 0.0406, 0.0124, -0.007, 0.0215, -0.0032, 0.0424, -0.0734, -0.0745, 0.0547, 0.0189, 0.0513, 0.0609, -0.0453, -0.0234, -0.0398, -0.0523, 0.0498, 0.0452, 0.0313, -0.0321, -0.0206, 0.0416, 0.0339, -0.0426, 0.0659, 0.0622, 0.0231, -0.0346, 0.0187, 0.0304, -0.0207, 0.0006, 0.0605, -0.0262, 0.0177, -0.0137, 0.0643, 0.0232, 0.0223, 0.0101, -0.0138, 0.0093, -0.0073, -0.0497, -0.0179, -0.0405, 0.0468, -0.026, 0.0216, -0.0453, 0.0534, -0.0317, -0.044, 0.0112, 0.0019, 0.0028, 0.0238, 0.0566, -0.0454, -0.0383, -0.0269, -0.0445, 0.0311, 0.0144, -0.0182, -0.0365, -0.015, -0.0265}, {0.0101, -0.0104, 0.0411, 0.0527, -0.0685, -0.0088, 0.076, -0.048, -0.0326, 0.0826, 0.0185, 0.0411, -0.0104, 0.0265, 0.0118, -0.0725, -0.0262, 0.0692, 0.0044, -0.0576, -0.0714, 0.0609, 0.0355, 0.048, 0.0228, -0.0176, -0.0648, 0.0107, -0.0494, 0.0554, 0.0268, -0.0419, -0.0529, -0.026, 0.0057, 0.0445, 0.0114, -0.0527, 0.0413, 0.026, -0.0517, 0.0292, -0.0369, -0.07, 0.0101, -0.0793, 0.0414, 0.0558, -0.061, 0.024, -0.0817, -0.0778, 0.055, -0.0367, -0.077, 0.0323, 0.0252, -0.0409, 0.0421, -0.0367, -0.0359, -0.0464, 0.0267, -0.0005, 0.0522, -0.0508, -0.0217, 0.0338, 0.0428, -0.0646, 0.0346, 0.0009, -0.0415, -0.006, 0.0363, 0.0377, -0.0263, 0.0209, 0.0836, -0.0591, 0.0727, 0.0597, 0.0585, -0.0224, 0.0222, 0.0437, -0.0315, 0.0528, -0.0453, -0.0852, 0.0223, 0.0104, -0.0403, -0.0445, -0.0794, 0.0448, 0.0368, -0.0042, 0.0403, 0.0262, 0.0173, 0.0457, -0.0067, -0.0414, -0.0181, -0.0621, 0.0519, 0.0143, 0.0194, 0.062, 0.0452, 0.0011, -0.0111, 0.0104, -0.0232, 0.0266, 0.0243, 0.0588, 0.0197, 0.0044, -0.0194, -0.0763, 0.0266, -0.0054, -0.0136, -0.0005, 0.0332, 0.0157, 0.0162, 0.0393, -0.0784, 0.043, 0.0178, -0.0086, -0.0096, 0.0304, -0.0184, 0.0228, -0.0906, 0.0419, -0.0045, -0.006, 0.0076, -0.0267, 0.0086, 0.0621, 0.0668, 0.0014, -0.0071, -0.0608, -0.0785, -0.0194, 0.0027, -0.0258, -0.0677, 0.03, 0.0036, 0.033, -0.0241, 0.0029}, {-0.0467, -0.0392, 0.0388, -0.0496, -0.0137, -0.0528, 0.0018, 0.0599, -0.046, 0.0061, 0.0054, 0.0219, 0.0293, -0.0193, 0.0623, -0.0159, -0.0453, 0.0539, 0.0586, -0.0195, -0.034, 0.0618, 0.0334, 0.014, -0.0402, 0.0243, 0.0648, -0.0574, -0.0053, 0.0325, -0.0199, 0.0141, 0.0191, -0.0633, -0.0363, 0.0469, -0.0383, -0.0194, 0.0806, -0.0149, -0.061, 0.0088, 0.0422, -0.081, 0.0346, 0.074, 0.0256, -0.0001, 0.0001, 0.0782, -0.0012, 0.0303, -0.0535, -0.0477, -0.0757, -0.0846, 0.043, -0.0093, -0.0479, 0.0241, -0.048, -0.0487, -0.0298, -0.0043, 0.0013, 0.0343, -0.0098, -0.0588, 0.0005, 0.0471, 0.014, 0.0595, 0.0285, -0.0402, -0.0023, 0.0526, 0.0358, 0.0286, 0.01, -0.0198, -0.0197, -0.0097, 0.0352, -0.053, -0.0033, 0.0292, -0.029, -0.0087, 0.0326, 0.0584, -0.047, -0.0261, 0.0375, 0.0531, -0.0124, 0.0547, 0.0222, 0.0332, 0.0533, 0.0561, -0.0209, 0.0542, -0.0178, -0.047, 0.0013, 0.0148, 0.0135, -0.0043, 0.0448, 0.0644, 0.0503, -0.0274, 0.0311, 0.0703, 0.0288, 0.0515, -0.0504, 0.0527, -0.0307, 0.0221, -0.0364, 0.0229, -0.0606, -0.0072, 0.078, 0.0505, 0.0837, -0.0268, 0.0216, 0.0481, 0.027, 0.0676, -0.0178, -0.0153, 0.0712, -0.056, 0.0579, -0.0442, 0.022, 0.0389, 0.0704, 0.0355, -0.006, 0.0137, -0.0571, -0.0204, -0.0395, 0.0341, -0.0409, -0.0215, -0.0321, -0.0711, 0.0586, 0.0555, -0.0603, 0.0121, 0.0157, -0.0514, -0.0201, 0.0158}, {-0.0562, -0.0002, -0.0745, -0.0246, 0.077, 0.0224, -0.0561, -0.0591, -0.0474, -0.0401, 0.0448, 0.0272, -0.0398, -0.0104, -0.0683, -0.0571, -0.0456, 0.0114, -0.0739, -0.0662, -0.0055, 0.0707, 0.0778, -0.0051, 0.0456, -0.0492, -0.0026, -0.0003, -0.0769, 0.0236, -0.0744, 0.0175, 0.032, -0.0161, 0.0221, 0.0277, 0.0221, -0.0633, 0.066, 0.0601, -0.0375, -0.0225, -0.0175, -0.0684, 0.0334, -0.0178, -0.0165, 0.0592, -0.0082, 0.013, -0.0723, -0.0313, -0.0461, 0.0684, -0.0582, 0.0487, 0.0638, 0.0754, -0.0189, 0.0548, -0.0027, 0.0056, -0.049, -0.0073, 0.0209, -0.0189, -0.0612, -0.0238, 0.0367, -0.0765, 0.005, -0.0398, -0.0176, -0.0125, -0.0209, -0.0135, -0.0812, -0.0807, -0.0717, 0.0387, 0.0543, 0.004, -0.0387, 0.012, 0.0266, 0.0569, -0.0743, 0.065, -0.0256, -0.0058, -0.0743, 0.056, 0.0569, 0.0492, 0.0068, -0.0423, 0.0443, 0.0485, 0.0379, -0.0118, 0.0646, 0.0626, 0.0489, -0.0037, -0.0392, -0.0778, -0.0705, 0.0496, -0.0488, -0.0284, 0.0645, -0.0596, 0.0099, -0.0289, -0.0333, 0.0314, 0.06, 0.0283, 0.0314, 0.0432, -0.05, -0.052, 0.0775, -0.0106, -0.0091, 0.0162, 0.0685, 0.0638, 0.0775, -0.0085, 0.0451, -0.0012, -0.0459, -0.0597, -0.0168, -0.0288, 0.0073, -0.0445, 0.0781, 0.0459, -0.008, 0.0327, -0.0793, -0.01, -0.0148, -0.0756, -0.0163, -0.0108, -0.028, -0.0353, -0.0791, -0.0715, -0.0046, -0.0755, 0.0429, -0.0199, 0.0587, -0.0724, 0.0035, 0.0664}};
static float linear_1_bias[LINEAR_1_C_OUT] = {-0.06, -0.0583, -0.0792, 0.0404};
