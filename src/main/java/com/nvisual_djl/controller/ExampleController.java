package com.nvisual_djl.controller;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.modality.cv.translator.ObjectDetectionTranslator;
import ai.djl.modality.cv.translator.SingleShotDetectionTranslator;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.nn.Block;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.*;
import ai.djl.util.JsonUtils;
import ai.djl.util.Utils;
import com.google.gson.annotations.SerializedName;
import com.nvisual_djl.services.ExamplesService;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author DanielTog
 * 2023/10/9
 */
@RestController
@RequestMapping("/example")
public class ExampleController {

    private static final ExamplesService examplesService = new ExamplesService();


    @Value("${dist.path}")
    static String distPath;

    @GetMapping("/predict_digit")
    public String predictImage() throws IOException {

        Path imageFile = Paths.get("input/digit_recognition/0.png");
        Image img = ImageFactory.getInstance().fromFile(imageFile);

        String modelName = "mlp";
        try (Model model = Model.newInstance(modelName)) {
            model.setBlock(new Mlp(28 * 28, 10, new int[] {128, 64}));

            // Assume you have run TrainMnist.java example, and saved model in build/model folder.
            Path modelDir = Paths.get("model");
            model.load(modelDir);

            List<String> classes =
                    IntStream.range(0, 10).mapToObj(String::valueOf).collect(Collectors.toList());
            Translator<ai.djl.modality.cv.Image, Classifications> translator =
                    ImageClassificationTranslator.builder()
                            .addTransform(new ToTensor())
                            .optSynset(classes)
                            .optApplySoftmax(true)
                            .build();

            try (Predictor<Image, Classifications> predictor = model.newPredictor(translator)) {
                return predictor.predict(img).toString();
            }
        } catch (MalformedModelException | TranslateException e) {
            throw new RuntimeException(e);
        }

    }

    @GetMapping("/predict_object")
    public static String predict() throws IOException, ModelException, TranslateException {
        Path imageFile = Paths.get("input/object_recognition/fruits.jpeg");
        Image img = ImageFactory.getInstance().fromFile(imageFile);

        String backbone;
        if ("TensorFlow".equals(Engine.getDefaultEngineName())) {
            backbone = "mobilenet_v2";
        } else {
            backbone = "resnet50";
        }



        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Image.class, DetectedObjects.class)
                        .optFilter("backbone", backbone)
                        .optEngine(Engine.getDefaultEngineName())
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<Image, DetectedObjects> model = criteria.loadModel()) {
            try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
                DetectedObjects detection = predictor.predict(img);
                examplesService.saveBoundingBoxImage(img, detection);
                return detection.toString();
            }
        }
    }

    @GetMapping("/predict_pikachu")
    public static int predictPikachu()
            throws IOException, MalformedModelException, TranslateException {

        String outputDir = "model";
        String imageFile = "input/pikachu_recognition/pikachu4.jpeg";

        try (Model model = Model.newInstance("pikachu-ssd")) {
            float detectionThreshold = 0.6f;
            // load parameters back to original training block
            model.setBlock(examplesService.getSsdTrainBlock());
            model.load(Paths.get(outputDir));
            // append prediction logic at end of training block with parameter loaded
            Block ssdTrain = model.getBlock();
            model.setBlock(examplesService.getSsdPredictBlock(ssdTrain));
            Path imagePath = Paths.get(imageFile);
            SingleShotDetectionTranslator translator =
                    SingleShotDetectionTranslator.builder()
                            .addTransform(new ToTensor())
                            .optSynset(Collections.singletonList("pikachu"))
                            .optThreshold(detectionThreshold)
                            .build();
            try (Predictor<Image, DetectedObjects> predictor = model.newPredictor(translator)) {
                Image image = ImageFactory.getInstance().fromFile(imagePath);
                DetectedObjects detectedObjects = predictor.predict(image);
                image.drawBoundingBoxes(detectedObjects);
                Path out = Paths.get(outputDir).resolve("pikachu_output.png");
                image.save(Files.newOutputStream(out), "png");
                // return number of pikachu detected
                return detectedObjects.getNumberOfObjects();
            }
        }
    }

    @GetMapping("/predict_custom_object")
    public static String predictCustomObject() throws IOException, ModelException, TranslateException {
        Path imageFile = Paths.get("input/object_recognition/ports_for_training_devices-2-25-2.png");
        Image img = ImageFactory.getInstance().fromFile(imageFile);


//        String modelUrl =  "https://developers.google.com/mediapipe/solutions/vision/object_detector#:~:text=EfficientDet%2DLite2%20(float%2032)";

//        String modelUrl = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz";

//        String modelUrl = "https://storage.googleapis.com/tf_model_garden/vision/qat/mobilenetv2_ssd_coco/mobilenetv2_ssd_i320_ckpt.tar.gz";

//        String modelUrl = "https://storage.googleapis.com/tf_model_garden/vision/qat/mobilenetv2_ssd_coco/mobilenetv2_ssd_i256_ckpt.tar.gz";

//        String modelUrl = "https://storage.googleapis.com/tf_model_garden/vision/qat/mobilenetv3.5_ssd_coco/mobilenetv3.5_ssd_i256_ckpt.tar.gz";

//        String modelUrl = "https://storage.googleapis.com/tf_model_garden/vision/qat/mobilenetv2_ssd_coco/mobilenetv3.5_ssd_i384_ckpt.tar.gz";

        String modelPath = "/Users/mauriziotognella/.djl.ai/cache/repo/model/undefined/ai/djl/localmodelzoo/d761c0c0b732b9afa5781d843fb0ee6ce1dfeaea/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model";

//        String modelPath = distPath + "/" + "saved_model";

        Path path = Paths.get(modelPath);

        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Image.class, DetectedObjects.class)
//                        .optModelUrls(modelUrl)
                        .optModelPath(path)
//                        .optFilter("backbone", "mobilenet_v2")
                        // saved_model.pb file is in the subfolder of the model archive file
                        .optModelName("saved_model")
//                        .optTranslator(new MyTranslator())
                        .optEngine("TensorFlow")
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<Image, DetectedObjects> model = criteria.loadModel();
             Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
            DetectedObjects detection = predictor.predict(img);
            examplesService.saveBoundingBoxImage(img, detection);
            return detection.toString();
        }
    }

    @GetMapping("/predict_custom_object_onnx")
    public static String predictCustomObjectOnnx() throws IOException, ModelException, TranslateException {
        Path imageFile = Paths.get("input/object_recognition/device4.jpg");
        Image img = ImageFactory.getInstance().fromFile(imageFile);

        String modeDir = "onnx_custom_model";

        Path modelPath = Paths.get(modeDir);

//        String modelPath = distPath + "/" + "saved_model";

        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .optModelPath(modelPath)
                        .setTypes(Image.class, DetectedObjects.class)
                        .optEngine("OnnxRuntime")
                        .optTranslator(new MyTranslator())
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<Image, DetectedObjects> model = criteria.loadModel();
             Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
            DetectedObjects detection = predictor.predict(img);
            examplesService.saveBoundingBoxImage(img, detection);
            return detection.toString();
        }
    }

    public static final class MyTranslator
            implements NoBatchifyTranslator<Image, DetectedObjects> {

        private Map<Integer, String> classes;
        private int maxBoxes;
        private float threshold;

        MyTranslator() {
            maxBoxes = 10;
            threshold = 0.7f;
        }

        /**
         * {@inheritDoc}
         */
//        @Override
//        public NDList processInput(TranslatorContext ctx, Image input) {
//            // input to tf object-detection models is a list of tensors, hence NDList
//            NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
//            Transform transform = new Resize(640);
//            array = transform.transform(array);
//            // tf object-detection models expect 8 bit unsigned integer tensor
//            array = array.toType(DataType.FLOAT32, true);
//            array = array.expandDims(0); // tf object-detection models expect a 4 dimensional input
//            return new NDList(array);
//        }

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, Image input) {
            // input to tf object-detection models is a list of tensors, hence NDList
            NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
            // optionally resize the image for faster processing
            array = NDImageUtils.resize(array, 640);
            // tf object-detection models expect 8 bit unsigned integer tensor
            array = array.toType(DataType.FLOAT32, true);
            array = array.expandDims(0); // tf object-detection models expect a 4 dimensional input
            return new NDList(array);
        }

        /** {@inheritDoc} */
//        @Override
//        public void prepare(TranslatorContext ctx) throws IOException {
//            if (classes == null) {
//                classes = examplesService.loadSynset();
//            }
//        }

        @Override
        public void prepare(TranslatorContext ctx) throws IOException {
            if (classes == null) {
                Path path = Paths.get("onnx_custom_model/synset.txt");
                List<String> list = Utils.readLines(path);

                int count = 1;
                Map<Integer, String > tempClasses = new HashMap<>();
                for(String row : list){
                    tempClasses.put(count, row);
                    count++;
                }

                classes = tempClasses;

                System.out.println(classes);
            }
        }

        /** {@inheritDoc} */
        @Override
        public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
            // output of tf object-detection models is a list of tensors, hence NDList in djl
            // output NDArray order in the list are not guaranteed

            int[] classIds = null;
            float[] probabilities = null;
            NDArray boundingBoxes = null;
            for (NDArray array : list) {
                if ("detection_boxes".equals(array.getName())) {
                    boundingBoxes = array.get(0);
                } else if ("detection_scores".equals(array.getName())) {
                    probabilities = array.get(0).toFloatArray();
                } else if ("detection_classes".equals(array.getName())) {
                    // class id is between 1 - number of classes
                    classIds = array.get(0).toType(DataType.INT32, true).toIntArray();
                }
            }
            Objects.requireNonNull(classIds);
            Objects.requireNonNull(probabilities);
            Objects.requireNonNull(boundingBoxes);

            List<String> retNames = new ArrayList<>();
            List<Double> retProbs = new ArrayList<>();
            List<BoundingBox> retBB = new ArrayList<>();

            // result are already sorted
            for (int i = 0; i < Math.min(classIds.length, maxBoxes); ++i) {
                int classId = classIds[i];
                double probability = probabilities[i];
                // classId starts from 1, -1 means background
                if (classId > 0 && probability > threshold) {
                    String className = classes.getOrDefault(classId, "#" + classId);
                    float[] box = boundingBoxes.get(i).toFloatArray();
                    float yMin = box[0];
                    float xMin = box[1];
                    float yMax = box[2];
                    float xMax = box[3];
                    Rectangle rect = new Rectangle(xMin, yMin, xMax - xMin, yMax - yMin);
                    retNames.add(className);
                    retProbs.add(probability);
                    retBB.add(rect);
                }
            }

            return new DetectedObjects(retNames, retProbs, retBB);
        }
    }

}
