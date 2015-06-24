import matplotlib
import matplotlib.pyplot as plt
import numpy as np

cube = [
    (0.4549019608,0.0000000000,0.5058823529),
    (0.4627450980,0.0000000000,0.5215686275),
    (0.4666666667,0.0000000000,0.5333333333),
    (0.4705882353,0.0000000000,0.5450980392),
    (0.4745098039,0.0000000000,0.5568627451),
    (0.4784313725,0.0000000000,0.5686274510),
    (0.4823529412,0.0000000000,0.5803921569),
    (0.4862745098,0.0000000000,0.5921568627),
    (0.4901960784,0.0000000000,0.6039215686),
    (0.4941176471,0.0000000000,0.6156862745),
    (0.4980392157,0.0039215686,0.6274509804),
    (0.5019607843,0.0078431373,0.6392156863),
    (0.5019607843,0.0156862745,0.6509803922),
    (0.5058823529,0.0235294118,0.6627450980),
    (0.5098039216,0.0313725490,0.6745098039),
    (0.5137254902,0.0431372549,0.6901960784),
    (0.5137254902,0.0509803922,0.7019607843),
    (0.5176470588,0.0627450980,0.7137254902),
    (0.5215686275,0.0705882353,0.7254901961),
    (0.5215686275,0.0784313725,0.7372549020),
    (0.5215686275,0.0901960784,0.7490196078),
    (0.5215686275,0.1019607843,0.7607843137),
    (0.5215686275,0.1137254902,0.7725490196),
    (0.5215686275,0.1254901961,0.7843137255),
    (0.5215686275,0.1372549020,0.7960784314),
    (0.5215686275,0.1490196078,0.8039215686),
    (0.5176470588,0.1607843137,0.8156862745),
    (0.5137254902,0.1725490196,0.8235294118),
    (0.5137254902,0.1843137255,0.8313725490),
    (0.5098039216,0.1960784314,0.8392156863),
    (0.5058823529,0.2078431373,0.8470588235),
    (0.5019607843,0.2196078431,0.8549019608),
    (0.5019607843,0.2313725490,0.8627450980),
    (0.4980392157,0.2431372549,0.8705882353),
    (0.4941176471,0.2509803922,0.8784313725),
    (0.4941176471,0.2588235294,0.8862745098),
    (0.4901960784,0.2705882353,0.8941176471),
    (0.4862745098,0.2784313725,0.9058823529),
    (0.4862745098,0.2862745098,0.9137254902),
    (0.4823529412,0.2941176471,0.9215686275),
    (0.4745098039,0.3019607843,0.9294117647),
    (0.4705882353,0.3098039216,0.9411764706),
    (0.4666666667,0.3176470588,0.9490196078),
    (0.4627450980,0.3254901961,0.9568627451),
    (0.4588235294,0.3333333333,0.9647058824),
    (0.4549019608,0.3411764706,0.9686274510),
    (0.4509803922,0.3490196078,0.9764705882),
    (0.4470588235,0.3568627451,0.9803921569),
    (0.4431372549,0.3647058824,0.9882352941),
    (0.4392156863,0.3686274510,0.9882352941),
    (0.4352941176,0.3764705882,0.9921568627),
    (0.4313725490,0.3843137255,0.9921568627),
    (0.4274509804,0.3921568627,0.9921568627),
    (0.4235294118,0.4000000000,0.9921568627),
    (0.4196078431,0.4078431373,0.9921568627),
    (0.4156862745,0.4196078431,0.9882352941),
    (0.4117647059,0.4274509804,0.9882352941),
    (0.4078431373,0.4352941176,0.9843137255),
    (0.4039215686,0.4431372549,0.9843137255),
    (0.4000000000,0.4509803922,0.9803921569),
    (0.4000000000,0.4588235294,0.9764705882),
    (0.3960784314,0.4666666667,0.9725490196),
    (0.3921568627,0.4745098039,0.9686274510),
    (0.3882352941,0.4823529412,0.9686274510),
    (0.3843137255,0.4901960784,0.9647058824),
    (0.3803921569,0.4941176471,0.9607843137),
    (0.3764705882,0.5019607843,0.9568627451),
    (0.3725490196,0.5098039216,0.9529411765),
    (0.3686274510,0.5176470588,0.9490196078),
    (0.3647058824,0.5254901961,0.9450980392),
    (0.3607843137,0.5294117647,0.9411764706),
    (0.3568627451,0.5372549020,0.9372549020),
    (0.3529411765,0.5411764706,0.9333333333),
    (0.3490196078,0.5490196078,0.9254901961),
    (0.3450980392,0.5568627451,0.9215686275),
    (0.3411764706,0.5607843137,0.9176470588),
    (0.3372549020,0.5686274510,0.9098039216),
    (0.3333333333,0.5725490196,0.9019607843),
    (0.3254901961,0.5803921569,0.8980392157),
    (0.3215686275,0.5843137255,0.8901960784),
    (0.3176470588,0.5921568627,0.8862745098),
    (0.3137254902,0.5960784314,0.8784313725),
    (0.3098039216,0.6000000000,0.8705882353),
    (0.3058823529,0.6078431373,0.8666666667),
    (0.3019607843,0.6117647059,0.8588235294),
    (0.2980392157,0.6196078431,0.8509803922),
    (0.2901960784,0.6235294118,0.8431372549),
    (0.2862745098,0.6313725490,0.8392156863),
    (0.2823529412,0.6352941176,0.8313725490),
    (0.2745098039,0.6431372549,0.8235294118),
    (0.2705882353,0.6470588235,0.8156862745),
    (0.2627450980,0.6549019608,0.8078431373),
    (0.2588235294,0.6588235294,0.8000000000),
    (0.2509803922,0.6666666667,0.7921568627),
    (0.2470588235,0.6705882353,0.7882352941),
    (0.2392156863,0.6784313725,0.7803921569),
    (0.2352941176,0.6823529412,0.7725490196),
    (0.2313725490,0.6862745098,0.7607843137),
    (0.2274509804,0.6941176471,0.7529411765),
    (0.2235294118,0.6980392157,0.7450980392),
    (0.2235294118,0.7019607843,0.7372549020),
    (0.2196078431,0.7098039216,0.7294117647),
    (0.2196078431,0.7137254902,0.7215686275),
    (0.2196078431,0.7176470588,0.7137254902),
    (0.2196078431,0.7215686275,0.7058823529),
    (0.2235294118,0.7254901961,0.6980392157),
    (0.2235294118,0.7333333333,0.6901960784),
    (0.2274509804,0.7372549020,0.6823529412),
    (0.2274509804,0.7411764706,0.6705882353),
    (0.2313725490,0.7450980392,0.6627450980),
    (0.2352941176,0.7490196078,0.6549019608),
    (0.2392156863,0.7529411765,0.6470588235),
    (0.2431372549,0.7568627451,0.6392156863),
    (0.2431372549,0.7607843137,0.6274509804),
    (0.2470588235,0.7647058824,0.6196078431),
    (0.2509803922,0.7686274510,0.6117647059),
    (0.2549019608,0.7725490196,0.6039215686),
    (0.2588235294,0.7764705882,0.5921568627),
    (0.2627450980,0.7803921569,0.5843137255),
    (0.2627450980,0.7843137255,0.5764705882),
    (0.2666666667,0.7882352941,0.5686274510),
    (0.2705882353,0.7921568627,0.5568627451),
    (0.2705882353,0.7960784314,0.5490196078),
    (0.2745098039,0.7960784314,0.5411764706),
    (0.2784313725,0.8000000000,0.5294117647),
    (0.2784313725,0.8039215686,0.5215686275),
    (0.2823529412,0.8078431373,0.5137254902),
    (0.2862745098,0.8117647059,0.5058823529),
    (0.2862745098,0.8156862745,0.4941176471),
    (0.2901960784,0.8156862745,0.4862745098),
    (0.2941176471,0.8196078431,0.4784313725),
    (0.2941176471,0.8235294118,0.4666666667),
    (0.2980392157,0.8274509804,0.4588235294),
    (0.3019607843,0.8274509804,0.4509803922),
    (0.3019607843,0.8313725490,0.4431372549),
    (0.3058823529,0.8352941176,0.4352941176),
    (0.3098039216,0.8392156863,0.4235294118),
    (0.3137254902,0.8431372549,0.4156862745),
    (0.3137254902,0.8470588235,0.4078431373),
    (0.3176470588,0.8470588235,0.3960784314),
    (0.3215686275,0.8509803922,0.3843137255),
    (0.3215686275,0.8549019608,0.3764705882),
    (0.3254901961,0.8588235294,0.3647058824),
    (0.3294117647,0.8627450980,0.3529411765),
    (0.3294117647,0.8666666667,0.3411764706),
    (0.3333333333,0.8705882353,0.3333333333),
    (0.3372549020,0.8705882353,0.3215686275),
    (0.3411764706,0.8745098039,0.3137254902),
    (0.3411764706,0.8784313725,0.3058823529),
    (0.3450980392,0.8823529412,0.2980392157),
    (0.3490196078,0.8823529412,0.2941176471),
    (0.3529411765,0.8862745098,0.2901960784),
    (0.3568627451,0.8901960784,0.2862745098),
    (0.3607843137,0.8901960784,0.2862745098),
    (0.3686274510,0.8941176471,0.2862745098),
    (0.3725490196,0.8980392157,0.2862745098),
    (0.3803921569,0.8980392157,0.2862745098),
    (0.3882352941,0.9019607843,0.2901960784),
    (0.3960784314,0.9019607843,0.2901960784),
    (0.4078431373,0.9058823529,0.2901960784),
    (0.4156862745,0.9058823529,0.2941176471),
    (0.4274509804,0.9098039216,0.2941176471),
    (0.4352941176,0.9098039216,0.2980392157),
    (0.4470588235,0.9137254902,0.2980392157),
    (0.4588235294,0.9137254902,0.3019607843),
    (0.4705882353,0.9176470588,0.3058823529),
    (0.4784313725,0.9176470588,0.3058823529),
    (0.4901960784,0.9176470588,0.3098039216),
    (0.5019607843,0.9215686275,0.3098039216),
    (0.5098039216,0.9215686275,0.3137254902),
    (0.5215686275,0.9215686275,0.3137254902),
    (0.5294117647,0.9215686275,0.3137254902),
    (0.5372549020,0.9215686275,0.3176470588),
    (0.5490196078,0.9215686275,0.3176470588),
    (0.5568627451,0.9215686275,0.3215686275),
    (0.5686274510,0.9215686275,0.3215686275),
    (0.5764705882,0.9215686275,0.3215686275),
    (0.5882352941,0.9254901961,0.3254901961),
    (0.5960784314,0.9254901961,0.3254901961),
    (0.6078431373,0.9254901961,0.3294117647),
    (0.6156862745,0.9254901961,0.3294117647),
    (0.6274509804,0.9254901961,0.3294117647),
    (0.6352941176,0.9254901961,0.3333333333),
    (0.6470588235,0.9254901961,0.3333333333),
    (0.6549019608,0.9254901961,0.3333333333),
    (0.6627450980,0.9254901961,0.3372549020),
    (0.6705882353,0.9254901961,0.3372549020),
    (0.6784313725,0.9254901961,0.3372549020),
    (0.6862745098,0.9254901961,0.3411764706),
    (0.6941176471,0.9254901961,0.3411764706),
    (0.7058823529,0.9254901961,0.3411764706),
    (0.7137254902,0.9254901961,0.3411764706),
    (0.7215686275,0.9254901961,0.3450980392),
    (0.7254901961,0.9254901961,0.3450980392),
    (0.7333333333,0.9254901961,0.3450980392),
    (0.7411764706,0.9254901961,0.3450980392),
    (0.7490196078,0.9254901961,0.3490196078),
    (0.7568627451,0.9254901961,0.3490196078),
    (0.7647058824,0.9254901961,0.3490196078),
    (0.7686274510,0.9254901961,0.3490196078),
    (0.7764705882,0.9254901961,0.3490196078),
    (0.7843137255,0.9254901961,0.3490196078),
    (0.7882352941,0.9254901961,0.3529411765),
    (0.7960784314,0.9254901961,0.3529411765),
    (0.8000000000,0.9254901961,0.3529411765),
    (0.8039215686,0.9254901961,0.3529411765),
    (0.8117647059,0.9254901961,0.3529411765),
    (0.8156862745,0.9215686275,0.3529411765),
    (0.8196078431,0.9176470588,0.3568627451),
    (0.8235294118,0.9176470588,0.3568627451),
    (0.8274509804,0.9137254902,0.3568627451),
    (0.8313725490,0.9098039216,0.3568627451),
    (0.8352941176,0.9019607843,0.3568627451),
    (0.8392156863,0.8980392157,0.3568627451),
    (0.8431372549,0.8941176471,0.3568627451),
    (0.8470588235,0.8862745098,0.3568627451),
    (0.8509803922,0.8823529412,0.3568627451),
    (0.8549019608,0.8784313725,0.3607843137),
    (0.8588235294,0.8705882353,0.3607843137),
    (0.8627450980,0.8666666667,0.3607843137),
    (0.8666666667,0.8588235294,0.3607843137),
    (0.8705882353,0.8549019608,0.3607843137),
    (0.8745098039,0.8509803922,0.3607843137),
    (0.8784313725,0.8431372549,0.3607843137),
    (0.8862745098,0.8392156863,0.3607843137),
    (0.8901960784,0.8352941176,0.3647058824),
    (0.8980392157,0.8274509804,0.3647058824),
    (0.9019607843,0.8235294118,0.3647058824),
    (0.9058823529,0.8156862745,0.3647058824),
    (0.9137254902,0.8078431373,0.3647058824),
    (0.9176470588,0.8039215686,0.3647058824),
    (0.9254901961,0.7960784314,0.3647058824),
    (0.9294117647,0.7882352941,0.3686274510),
    (0.9333333333,0.7843137255,0.3686274510),
    (0.9372549020,0.7764705882,0.3686274510),
    (0.9411764706,0.7686274510,0.3686274510),
    (0.9450980392,0.7607843137,0.3686274510),
    (0.9490196078,0.7529411765,0.3686274510),
    (0.9529411765,0.7450980392,0.3686274510),
    (0.9529411765,0.7372549020,0.3686274510),
    (0.9568627451,0.7294117647,0.3686274510),
    (0.9568627451,0.7215686275,0.3686274510),
    (0.9607843137,0.7137254902,0.3686274510),
    (0.9607843137,0.7058823529,0.3686274510),
    (0.9647058824,0.6980392157,0.3686274510),
    (0.9647058824,0.6901960784,0.3647058824),
    (0.9686274510,0.6784313725,0.3647058824),
    (0.9686274510,0.6705882353,0.3647058824),
    (0.9725490196,0.6588235294,0.3647058824),
    (0.9725490196,0.6509803922,0.3647058824),
    (0.9725490196,0.6392156863,0.3607843137),
    (0.9764705882,0.6313725490,0.3607843137),
    (0.9764705882,0.6196078431,0.3607843137),
    (0.9764705882,0.6117647059,0.3607843137),
    (0.9764705882,0.6000000000,0.3568627451),
    (0.9764705882,0.5882352941,0.3568627451)]

cubeYF = [
    (0.4810000000,0.0080000000,0.5640000000),
    (0.4840000000,0.0120000000,0.5740000000),
    (0.4870000000,0.0150000000,0.5830000000),
    (0.4910000000,0.0190000000,0.5920000000),
    (0.4940000000,0.0220000000,0.6010000000),
    (0.4970000000,0.0260000000,0.6100000000),
    (0.5010000000,0.0300000000,0.6190000000),
    (0.5040000000,0.0330000000,0.6280000000),
    (0.5070000000,0.0380000000,0.6370000000),
    (0.5100000000,0.0430000000,0.6470000000),
    (0.5120000000,0.0470000000,0.6580000000),
    (0.5150000000,0.0480000000,0.6700000000),
    (0.5180000000,0.0470000000,0.6830000000),
    (0.5210000000,0.0510000000,0.6930000000),
    (0.5230000000,0.0570000000,0.7020000000),
    (0.5250000000,0.0650000000,0.7110000000),
    (0.5260000000,0.0740000000,0.7200000000),
    (0.5260000000,0.0840000000,0.7290000000),
    (0.5270000000,0.0940000000,0.7370000000),
    (0.5270000000,0.1040000000,0.7450000000),
    (0.5270000000,0.1130000000,0.7540000000),
    (0.5260000000,0.1240000000,0.7610000000),
    (0.5250000000,0.1330000000,0.7690000000),
    (0.5240000000,0.1430000000,0.7770000000),
    (0.5220000000,0.1520000000,0.7860000000),
    (0.5210000000,0.1600000000,0.7940000000),
    (0.5210000000,0.1680000000,0.8020000000),
    (0.5200000000,0.1760000000,0.8080000000),
    (0.5190000000,0.1840000000,0.8170000000),
    (0.5170000000,0.1910000000,0.8250000000),
    (0.5160000000,0.1990000000,0.8320000000),
    (0.5150000000,0.2070000000,0.8380000000),
    (0.5130000000,0.2140000000,0.8460000000),
    (0.5110000000,0.2220000000,0.8530000000),
    (0.5090000000,0.2290000000,0.8600000000),
    (0.5070000000,0.2370000000,0.8670000000),
    (0.5040000000,0.2450000000,0.8730000000),
    (0.5020000000,0.2520000000,0.8800000000),
    (0.5000000000,0.2590000000,0.8870000000),
    (0.4990000000,0.2650000000,0.8930000000),
    (0.4970000000,0.2710000000,0.9010000000),
    (0.4950000000,0.2780000000,0.9090000000),
    (0.4920000000,0.2840000000,0.9160000000),
    (0.4880000000,0.2910000000,0.9230000000),
    (0.4790000000,0.3010000000,0.9320000000),
    (0.4740000000,0.3080000000,0.9390000000),
    (0.4710000000,0.3140000000,0.9450000000),
    (0.4670000000,0.3190000000,0.9500000000),
    (0.4630000000,0.3250000000,0.9570000000),
    (0.4600000000,0.3300000000,0.9620000000),
    (0.4570000000,0.3360000000,0.9670000000),
    (0.4530000000,0.3420000000,0.9740000000),
    (0.4500000000,0.3480000000,0.9790000000),
    (0.4470000000,0.3520000000,0.9830000000),
    (0.4450000000,0.3570000000,0.9860000000),
    (0.4420000000,0.3620000000,0.9880000000),
    (0.4380000000,0.3690000000,0.9880000000),
    (0.4340000000,0.3740000000,0.9890000000),
    (0.4320000000,0.3800000000,0.9910000000),
    (0.4300000000,0.3850000000,0.9920000000),
    (0.4270000000,0.3920000000,0.9930000000),
    (0.4250000000,0.3970000000,0.9940000000),
    (0.4220000000,0.4020000000,0.9960000000),
    (0.4180000000,0.4090000000,0.9960000000),
    (0.4150000000,0.4140000000,0.9980000000),
    (0.4120000000,0.4200000000,0.9990000000),
    (0.4100000000,0.4250000000,1.0000000000),
    (0.4080000000,0.4320000000,0.9990000000),
    (0.4080000000,0.4390000000,0.9970000000),
    (0.4070000000,0.4450000000,0.9950000000),
    (0.4050000000,0.4500000000,0.9930000000),
    (0.4020000000,0.4560000000,0.9910000000),
    (0.3990000000,0.4620000000,0.9880000000),
    (0.3960000000,0.4670000000,0.9840000000),
    (0.3920000000,0.4720000000,0.9800000000),
    (0.3890000000,0.4790000000,0.9780000000),
    (0.3860000000,0.4880000000,0.9720000000),
    (0.3820000000,0.4960000000,0.9670000000),
    (0.3780000000,0.5040000000,0.9610000000),
    (0.3760000000,0.5100000000,0.9560000000),
    (0.3710000000,0.5150000000,0.9520000000),
    (0.3680000000,0.5190000000,0.9480000000),
    (0.3650000000,0.5240000000,0.9460000000),
    (0.3620000000,0.5280000000,0.9420000000),
    (0.3580000000,0.5330000000,0.9380000000),
    (0.3550000000,0.5380000000,0.9330000000),
    (0.3520000000,0.5420000000,0.9280000000),
    (0.3470000000,0.5470000000,0.9240000000),
    (0.3440000000,0.5510000000,0.9190000000),
    (0.3410000000,0.5560000000,0.9150000000),
    (0.3380000000,0.5600000000,0.9100000000),
    (0.3340000000,0.5650000000,0.9060000000),
    (0.3310000000,0.5700000000,0.9010000000),
    (0.3280000000,0.5750000000,0.8960000000),
    (0.3230000000,0.5820000000,0.8890000000),
    (0.3200000000,0.5870000000,0.8840000000),
    (0.3170000000,0.5920000000,0.8790000000),
    (0.3140000000,0.5960000000,0.8740000000),
    (0.3090000000,0.6010000000,0.8700000000),
    (0.3070000000,0.6050000000,0.8650000000),
    (0.3040000000,0.6100000000,0.8610000000),
    (0.3000000000,0.6150000000,0.8560000000),
    (0.2960000000,0.6190000000,0.8520000000),
    (0.2930000000,0.6240000000,0.8470000000),
    (0.2890000000,0.6280000000,0.8420000000),
    (0.2860000000,0.6330000000,0.8380000000),
    (0.2830000000,0.6370000000,0.8330000000),
    (0.2790000000,0.6420000000,0.8290000000),
    (0.2750000000,0.6480000000,0.8220000000),
    (0.2690000000,0.6550000000,0.8140000000),
    (0.2630000000,0.6630000000,0.8030000000),
    (0.2600000000,0.6680000000,0.7960000000),
    (0.2570000000,0.6710000000,0.7900000000),
    (0.2530000000,0.6740000000,0.7850000000),
    (0.2490000000,0.6780000000,0.7790000000),
    (0.2460000000,0.6820000000,0.7730000000),
    (0.2430000000,0.6870000000,0.7680000000),
    (0.2390000000,0.6910000000,0.7630000000),
    (0.2360000000,0.6930000000,0.7560000000),
    (0.2340000000,0.6970000000,0.7510000000),
    (0.2310000000,0.7010000000,0.7450000000),
    (0.2280000000,0.7060000000,0.7390000000),
    (0.2240000000,0.7100000000,0.7330000000),
    (0.2210000000,0.7120000000,0.7260000000),
    (0.2190000000,0.7160000000,0.7200000000),
    (0.2170000000,0.7200000000,0.7150000000),
    (0.2170000000,0.7230000000,0.7090000000),
    (0.2190000000,0.7260000000,0.7020000000),
    (0.2220000000,0.7300000000,0.6960000000),
    (0.2240000000,0.7340000000,0.6890000000),
    (0.2260000000,0.7360000000,0.6850000000),
    (0.2290000000,0.7380000000,0.6790000000),
    (0.2310000000,0.7410000000,0.6720000000),
    (0.2330000000,0.7450000000,0.6650000000),
    (0.2350000000,0.7490000000,0.6580000000),
    (0.2380000000,0.7510000000,0.6520000000),
    (0.2400000000,0.7540000000,0.6450000000),
    (0.2420000000,0.7560000000,0.6380000000),
    (0.2450000000,0.7580000000,0.6310000000),
    (0.2470000000,0.7610000000,0.6250000000),
    (0.2490000000,0.7650000000,0.6190000000),
    (0.2520000000,0.7690000000,0.6130000000),
    (0.2560000000,0.7740000000,0.5990000000),
    (0.2600000000,0.7770000000,0.5900000000),
    (0.2620000000,0.7800000000,0.5820000000),
    (0.2640000000,0.7840000000,0.5750000000),
    (0.2670000000,0.7880000000,0.5680000000),
    (0.2690000000,0.7910000000,0.5620000000),
    (0.2710000000,0.7930000000,0.5550000000),
    (0.2740000000,0.7950000000,0.5480000000),
    (0.2740000000,0.7970000000,0.5410000000),
    (0.2750000000,0.8000000000,0.5340000000),
    (0.2760000000,0.8020000000,0.5270000000),
    (0.2790000000,0.8050000000,0.5210000000),
    (0.2810000000,0.8090000000,0.5140000000),
    (0.2830000000,0.8120000000,0.5070000000),
    (0.2860000000,0.8150000000,0.5000000000),
    (0.2880000000,0.8170000000,0.4930000000),
    (0.2900000000,0.8200000000,0.4860000000),
    (0.2920000000,0.8220000000,0.4800000000),
    (0.2950000000,0.8240000000,0.4720000000),
    (0.2970000000,0.8260000000,0.4630000000),
    (0.2990000000,0.8290000000,0.4550000000),
    (0.3020000000,0.8310000000,0.4480000000),
    (0.3040000000,0.8330000000,0.4410000000),
    (0.3060000000,0.8360000000,0.4350000000),
    (0.3080000000,0.8380000000,0.4280000000),
    (0.3110000000,0.8400000000,0.4210000000),
    (0.3130000000,0.8420000000,0.4140000000),
    (0.3150000000,0.8450000000,0.4070000000),
    (0.3180000000,0.8470000000,0.4010000000),
    (0.3200000000,0.8490000000,0.3960000000),
    (0.3220000000,0.8520000000,0.3910000000),
    (0.3240000000,0.8540000000,0.3840000000),
    (0.3300000000,0.8580000000,0.3730000000),
    (0.3350000000,0.8610000000,0.3610000000),
    (0.3390000000,0.8650000000,0.3520000000),
    (0.3410000000,0.8670000000,0.3450000000),
    (0.3440000000,0.8690000000,0.3400000000),
    (0.3460000000,0.8710000000,0.3350000000),
    (0.3480000000,0.8740000000,0.3280000000),
    (0.3490000000,0.8760000000,0.3230000000),
    (0.3500000000,0.8780000000,0.3180000000),
    (0.3510000000,0.8810000000,0.3130000000),
    (0.3530000000,0.8830000000,0.3090000000),
    (0.3560000000,0.8850000000,0.3040000000),
    (0.3580000000,0.8870000000,0.2980000000),
    (0.3600000000,0.8900000000,0.2920000000),
    (0.3630000000,0.8920000000,0.2850000000),
    (0.3660000000,0.8940000000,0.2800000000),
    (0.3720000000,0.8960000000,0.2780000000),
    (0.3780000000,0.8980000000,0.2790000000),
    (0.3830000000,0.8980000000,0.2810000000),
    (0.3890000000,0.9000000000,0.2820000000),
    (0.3950000000,0.9020000000,0.2830000000),
    (0.4020000000,0.9040000000,0.2840000000),
    (0.4090000000,0.9060000000,0.2870000000),
    (0.4160000000,0.9060000000,0.2890000000),
    (0.4230000000,0.9070000000,0.2910000000),
    (0.4290000000,0.9090000000,0.2930000000),
    (0.4360000000,0.9100000000,0.2960000000),
    (0.4430000000,0.9100000000,0.2980000000),
    (0.4500000000,0.9120000000,0.3000000000),
    (0.4570000000,0.9140000000,0.3020000000),
    (0.4640000000,0.9140000000,0.3020000000),
    (0.4720000000,0.9140000000,0.3030000000),
    (0.4830000000,0.9140000000,0.3060000000),
    (0.4950000000,0.9160000000,0.3100000000),
    (0.5100000000,0.9180000000,0.3140000000),
    (0.5170000000,0.9200000000,0.3140000000),
    (0.5250000000,0.9210000000,0.3150000000),
    (0.5340000000,0.9220000000,0.3170000000),
    (0.5420000000,0.9220000000,0.3190000000),
    (0.5490000000,0.9220000000,0.3210000000),
    (0.5580000000,0.9220000000,0.3220000000),
    (0.5670000000,0.9220000000,0.3220000000),
    (0.5760000000,0.9240000000,0.3240000000),
    (0.5850000000,0.9250000000,0.3270000000),
    (0.5940000000,0.9250000000,0.3280000000),
    (0.6030000000,0.9250000000,0.3290000000),
    (0.6120000000,0.9250000000,0.3300000000),
    (0.6220000000,0.9250000000,0.3320000000),
    (0.6310000000,0.9250000000,0.3330000000),
    (0.6390000000,0.9250000000,0.3330000000),
    (0.6430000000,0.9250000000,0.3330000000),
    (0.6450000000,0.9250000000,0.3340000000),
    (0.6520000000,0.9250000000,0.3350000000),
    (0.6610000000,0.9250000000,0.3370000000),
    (0.6700000000,0.9250000000,0.3370000000),
    (0.6730000000,0.9250000000,0.3370000000),
    (0.6770000000,0.9250000000,0.3380000000),
    (0.6810000000,0.9250000000,0.3390000000),
    (0.6910000000,0.9250000000,0.3410000000),
    (0.7000000000,0.9250000000,0.3410000000),
    (0.7050000000,0.9250000000,0.3410000000),
    (0.7070000000,0.9250000000,0.3410000000),
    (0.7120000000,0.9250000000,0.3430000000),
    (0.7180000000,0.9250000000,0.3440000000),
    (0.7220000000,0.9250000000,0.3450000000),
    (0.7230000000,0.9250000000,0.3450000000),
    (0.7320000000,0.9250000000,0.3450000000),
    (0.7410000000,0.9250000000,0.3460000000),
    (0.7490000000,0.9250000000,0.3480000000),
    (0.7530000000,0.9250000000,0.3490000000),
    (0.7550000000,0.9250000000,0.3490000000),
    (0.7620000000,0.9250000000,0.3490000000),
    (0.7710000000,0.9250000000,0.3500000000),
    (0.7800000000,0.9250000000,0.3520000000),
    (0.7830000000,0.9250000000,0.3530000000),
    (0.7870000000,0.9250000000,0.3530000000),
    (0.7920000000,0.9250000000,0.3530000000),
    (0.8000000000,0.9250000000,0.3530000000),
    (0.8000000000,0.9250000000,0.3530000000),
    (0.8050000000,0.9250000000,0.3540000000),
    (0.8120000000,0.9240000000,0.3560000000),
    (0.8190000000,0.9210000000,0.3570000000)]

def make_cmap(colors, position=None, bit=False):
    '''
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap


cubecmap = make_cmap(cube)
cubeYFcmap = make_cmap(cubeYF)
