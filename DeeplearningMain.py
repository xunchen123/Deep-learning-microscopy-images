from segment import SegmentMain
from detection import DetectionMain
from classification import ClassifyMain

# 0 CNN+W ; 1 CNN+U ; 2 CNN+WU
def ProDeeplearningMain(folder_test = '', folder_train = '', analysis_type = dict()):

    if analysis_type["analysis_type"] == 0:
        SegmentMain.segmentMain(folder_test, folder_train, analysis_type)
    elif analysis_type["analysis_type"] == 1:
        DetectionMain.detectionMain(folder_test, folder_train, analysis_type)
    elif analysis_type["analysis_type"] == 2:
        # detection + classify
        # DetectionMain.detectionMain(folder_test, folder_train, analysis_type)
        ClassifyMain.classifyMain(folder_test, folder_train, analysis_type)
    else:
        raise ValueError("Analysis Mapping Wrong")

    print("Test folder: ", folder_test)
    print("Analysis type: ", analysis_type)
    print("complete")