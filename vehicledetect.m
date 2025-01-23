% Load Pretrained YOLO v4 Object Detector
model = yolov4ObjectDetector('csp-darknet53-coco');

% Load an Image
image = imread('france-paris-street-scene-traffic-crowd-people-H44MDD.jpg');

% Run Object Detection
[bboxes, scores, labels] = detect(model, image);

% Define Vehicle Classes (COCO IDs)
vehicleClasses = {'car', 'motorbike', 'bus', 'truck'};

% Filter for Vehicle Detections
isVehicle = ismember(labels, vehicleClasses);
vehicleBboxes = bboxes(isVehicle, :);
vehicleScores = scores(isVehicle);
vehicleLabels = labels(isVehicle);

% Annotate Image with Bounding Boxes and Class Labels
annotatedImage = insertObjectAnnotation(image, 'rectangle', vehicleBboxes, cellstr(vehicleLabels), 'Color', 'green');

% Display Annotated Image
imshow(annotatedImage);
title(['Vehicle Count: ', num2str(sum(isVehicle))]);
