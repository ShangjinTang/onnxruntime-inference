package com.example.onnxinferinjava;

import android.util.Log;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.OnnxTensor;

import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Map;

public class ModelRunner {
    private static final String TAG = "ModelRunner";
    private InputStream mModelInputStream = null;
    private int mInputSize = 0;

    public ModelRunner(InputStream modelInputStream, int inputSize) throws Exception {

        mModelInputStream = modelInputStream;
        mInputSize = inputSize;
    }

    public long runInference(float[] inputData) throws Exception {

        try (OrtEnvironment mEnvironment = OrtEnvironment.getEnvironment()) {
            try (OrtSession.SessionOptions options = new SessionOptions()) {
                byte[] modelData = mModelInputStream.readAllBytes();
                try (OrtSession session = mEnvironment.createSession(modelData, options)) {
                    Log.i(TAG, "Inputs:");
                    for (NodeInfo i : session.getInputInfo().values()) {
                        Log.i(TAG, i.toString());
                    }

                    Log.i(TAG, "Outputs:");
                    for (NodeInfo i : session.getOutputInfo().values()) {
                        Log.i(TAG, i.toString());
                    }

                    // Get the name of the input node
                    String inputName = session.getInputNames().iterator().next();
                    // Make a FloatBuffer of the inputs
                    FloatBuffer floatBufferInputs = FloatBuffer.wrap(inputData);
                    // Create input tensor with floatBufferInputs of shape (1, 1)
                    try (OnnxTensor inputTensor = OnnxTensor.createTensor(mEnvironment, floatBufferInputs, new long[]{1, mInputSize});
                         Result results = session.run(Collections.singletonMap(inputName, inputTensor));) {
                        long[] output = (long[]) results.get(0).getValue();
                        Log.i(TAG, "Inference result: " + output[0]);
                        return output[0];
                    }
                }
        }


//            Log.i(TAG, "Model loaded");
        }
    }
}