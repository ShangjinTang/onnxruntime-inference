package com.example.onnxinferinjava;

import android.content.Context;
import android.util.Log;

public class XgbcOnnxModel extends OnnxModel {

    private static final String TAG = "XgbcOnnxModel";

    public XgbcOnnxModel(Context context, String modelFile) {
        super(context, modelFile);
    }

    public float[] runInferenceGetProbabilities(float[] inputData) {
        String probabilitiesOutputName = "output_probabilities";
        Object result = runInference(inputData, probabilitiesOutputName);

        if (result instanceof float[]) {
            return (float[]) result;
        } else {
            Log.e(TAG, "Unexpected output type for probabilities: " + (result != null ? result.getClass().getName() : "null"));
            return null;
        }
    }

    public Long runInferenceGetLabel(float[] inputData) {
        String labelOutputName = "output_label";
        Object result = runInference(inputData, labelOutputName);

        if (result instanceof Long) {
            return (Long) result;
        } else {
            Log.e(TAG, "Unexpected output type for label: " + (result != null ? result.getClass().getName() : "null"));
            return null;
        }
    }
}