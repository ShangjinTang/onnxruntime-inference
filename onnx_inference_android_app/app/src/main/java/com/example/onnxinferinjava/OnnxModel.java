package com.example.onnxinferinjava;

import android.content.Context;
import android.util.Log;

import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.OnnxTensor;

import java.io.InputStream;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

public class OnnxModel {
    public static final String TAG = "OnnxModel";

    private final Context mContext;

    private OrtEnvironment mOrtEnvironment;
    private OrtSession mOrtSession;
    private SessionOptions mSessionOptions;

    private final String mModelAssetPath;
    private InputStream mModelInputStream;
    private byte[] mModelData;

    public OnnxModel(Context context, String modelFile) {
        mContext = context;
        mModelAssetPath = modelFile;
    }

    public void init() {
        Log.d(TAG, "OnnxModel onOpenSession enter");
        try {
            mOrtEnvironment = OrtEnvironment.getEnvironment();
            mSessionOptions = new SessionOptions();

            mModelInputStream = mContext.getAssets().open(mModelAssetPath);
            mModelData = mModelInputStream.readAllBytes();

            mOrtSession = mOrtEnvironment.createSession(mModelData, mSessionOptions);
        } catch (OrtException e) {
            Log.e(TAG, "open session for " + mModelAssetPath + " error: " + e.getMessage(), e);
        } catch (IOException e) {
            Log.e(TAG, "open session IOException for " + mModelAssetPath + " error: " + e.getMessage(), e);
        } catch (Exception e) {
            Log.e(TAG, "open session exception " + e.getMessage(), e);
        } finally {
            if (mModelInputStream != null) {
                try {
                    mModelInputStream.close();
                } catch (IOException e) {
                    Log.e(TAG, "Error closing InputStream: " + e.getMessage(), e);
                }
            }
        }
    }


    public void deinit() {
        Log.d(TAG, "OnnxModel onCloseSession enter");
        try {
            if (mSessionOptions != null) {
                mSessionOptions.close();
                mSessionOptions = null;
            }
            if (mOrtSession != null) {
                mOrtSession.close();
                mOrtSession = null;
            }
        } catch (OrtException e) {
            Log.e(TAG, "close session for " + mModelAssetPath + " error: " + e.getMessage(), e);
        } finally {
            mOrtEnvironment = null;
            mModelData = null;
            mModelInputStream = null;
            Log.d(TAG, "OnnxModel onCloseSession complete");
        }
    }

    /**
     * Runs inference on the ONNX model and returns a map containing all outputs.
     * <p>
     * This method performs the inference using the provided input data and returns
     * a map where the keys are the output names and the values are the corresponding
     * output values.
     *
     * @param inputData The input data for the model.
     * @return A map containing all outputs, or null if inference fails. The keys
     *         of the map are the output names (Strings), and the values are the
     *         corresponding output values (Objects).  The specific type of the
     *         output value depends on the model definition.
     */
    public Map<String, Object> runInference(float[] inputData) {
        Log.d(TAG, "runInference() called with: inputData = [" + inputData + "]");

        if (mOrtSession == null) {
            Log.e(TAG, "OrtSession is null.");
            return null;
        }

        try {
            Iterator<String> inputNameIterator = mOrtSession.getInputNames().iterator();
            if (!inputNameIterator.hasNext()) {
                Log.e(TAG, "No input names found in ONNX model.");
                return null;
            }
            String inputName = inputNameIterator.next();

            FloatBuffer floatBufferInputs = FloatBuffer.wrap(inputData);
            OnnxTensor inputTensor = OnnxTensor.createTensor(mOrtEnvironment, floatBufferInputs, new long[]{1, inputData.length});

            Result results = mOrtSession.run(Collections.singletonMap(inputName, inputTensor));

            Map<String, Object> outputMap = new HashMap<>();

            Set<String> outputNames = mOrtSession.getOutputInfo().keySet();
            for (String outputName : outputNames) {
                Optional<?> outputObj = results.get(outputName);
                if (outputObj.isPresent()) {
                    Object value = ((OnnxValue) outputObj.get()).getValue();

                    if (value instanceof long[][]) {
                        long[][] long2DArray = (long[][]) value;
                        Log.d(TAG, "  Detected long[][]");
                        if (long2DArray.length > 0) {
                            Log.d(TAG, "  Extracted first long[]: " + java.util.Arrays.toString(long2DArray[0]));
                            outputMap.put(outputName, long2DArray[0]);
                        } else {
                            outputMap.put(outputName, value);
                        }
                    } else if (value instanceof float[][]) {
                        float[][] float2DArray = (float[][]) value;
                        Log.d(TAG, "  Detected float[][]");
                        if (float2DArray.length > 0) {
                            Log.d(TAG, "  Extracted first float[]: " + java.util.Arrays.toString(float2DArray[0]));
                            outputMap.put(outputName, float2DArray[0]);
                        } else {
                            outputMap.put(outputName, value);
                        }
                    } else if (value instanceof int[][]) {
                        int[][] int2DArray = (int[][]) value;
                        Log.d(TAG, "  Detected int[][]");
                        if (int2DArray.length > 0) {
                            Log.d(TAG, "  Extracted first int[]: " + java.util.Arrays.toString(int2DArray[0]));
                            outputMap.put(outputName, int2DArray[0]);
                        } else {
                            outputMap.put(outputName, value);
                        }
                    } else if (value instanceof double[][]) {
                        double[][] double2DArray = (double[][]) value;
                        Log.d(TAG, "  Detected double[][]");
                        if (double2DArray.length > 0) {
                            Log.d(TAG, "  Extracted first double[]: " + java.util.Arrays.toString(double2DArray[0]));
                            outputMap.put(outputName, double2DArray[0]);
                        } else {
                            outputMap.put(outputName, value);
                        }
                    } else if (value instanceof byte[][]) {
                        byte[][] byte2DArray = (byte[][]) value;
                        Log.d(TAG, "  Detected byte[][]");
                        if (byte2DArray.length > 0) {
                            Log.d(TAG, "  Extracted first byte[]: " + java.util.Arrays.toString(byte2DArray[0]));
                            outputMap.put(outputName, byte2DArray[0]);
                        } else {
                            outputMap.put(outputName, value);
                        }
                    } else if (value instanceof short[][]) {
                        short[][] short2DArray = (short[][]) value;
                        Log.d(TAG, "  Detected short[][]");
                        if (short2DArray.length > 0) {
                            Log.d(TAG, "  Extracted first short[]: " + java.util.Arrays.toString(short2DArray[0]));
                            outputMap.put(outputName, short2DArray[0]);
                        } else {
                            outputMap.put(outputName, value);
                        }
                    } else if (value instanceof boolean[][]) {
                        boolean[][] boolean2DArray = (boolean[][]) value;
                        Log.d(TAG, "  Detected boolean[][]");
                        if (boolean2DArray.length > 0) {
                            Log.d(TAG, "  Extracted first boolean[]: " + java.util.Arrays.toString(boolean2DArray[0]));
                            outputMap.put(outputName, boolean2DArray[0]);
                        } else {
                            outputMap.put(outputName, value);
                        }
                    } else if (value instanceof String[][]) {
                        String[][] string2DArray = (String[][]) value;
                        Log.d(TAG, "  Detected String[][]");
                        if (string2DArray.length > 0) {
                            Log.d(TAG, "  Extracted first String[]: " + java.util.Arrays.toString(string2DArray[0]));
                            outputMap.put(outputName, string2DArray[0]);
                        } else {
                            outputMap.put(outputName, value);
                        }
                    } else if (value instanceof long[]) {
                        long[] longArray = (long[]) value;
                        Log.d(TAG, "  Detected long[]");
                        if (longArray.length > 0) {
                            Log.d(TAG, "  Extracted first long: " + longArray[0]);
                            outputMap.put(outputName, longArray[0]);
                        } else {
                            outputMap.put(outputName, value);
                        }
                    } else if (value instanceof float[]) {
                        float[] floatArray = (float[]) value;
                        Log.d(TAG, "  Detected float[]");
                        if (floatArray.length > 0) {
                            Log.d(TAG, "  Extracted first float: " + floatArray[0]);
                            outputMap.put(outputName, floatArray[0]);
                        } else {
                            outputMap.put(outputName, value);
                        }
                    } else if (value instanceof int[]) {
                        int[] intArray = (int[]) value;
                        Log.d(TAG, "  Detected int[]");
                        if (intArray.length > 0) {
                            Log.d(TAG, "  Extracted first int: " + intArray[0]);
                            outputMap.put(outputName, intArray[0]);
                        } else {
                            outputMap.put(outputName, value);
                        }
                    } else if (value instanceof double[]) {
                        double[] doubleArray = (double[]) value;
                        Log.d(TAG, "  Detected double[]");
                        if (doubleArray.length > 0) {
                            Log.d(TAG, "  Extracted first double: " + doubleArray[0]);
                            outputMap.put(outputName, doubleArray[0]);
                        } else {
                            outputMap.put(outputName, value);
                        }
                    } else if (value instanceof byte[]) {
                        byte[] byteArray = (byte[]) value;
                        Log.d(TAG, "  Detected byte[]");
                        if (byteArray.length > 0) {
                            Log.d(TAG, "  Extracted first byte: " + byteArray[0]);
                            outputMap.put(outputName, byteArray[0]);
                        } else {
                            outputMap.put(outputName, value);
                        }
                    } else if (value instanceof short[]) {
                        short[] shortArray = (short[]) value;
                        Log.d(TAG, "  Detected short[]");
                        if (shortArray.length > 0) {
                            Log.d(TAG, "  Extracted first short: " + shortArray[0]);
                            outputMap.put(outputName, shortArray[0]);
                        } else {
                            outputMap.put(outputName, value);
                        }
                    } else if (value instanceof boolean[]) {
                        boolean[] booleanArray = (boolean[]) value;
                        Log.d(TAG, "  Detected boolean[]");
                        if (booleanArray.length > 0) {
                            Log.d(TAG, "  Extracted first boolean: " + booleanArray[0]);
                            outputMap.put(outputName, booleanArray[0]);
                        } else {
                            outputMap.put(outputName, value);
                        }
                    } else if (value instanceof String[]) {
                        String[] stringArray = (String[]) value;
                        Log.d(TAG, "  Detected String[]");
                        if (stringArray.length > 0) {
                            Log.d(TAG, "  Extracted first String: " + stringArray[0]);
                            outputMap.put(outputName, stringArray[0]);
                        } else {
                            outputMap.put(outputName, value);
                        }
                    } else {
                        Log.w(TAG, "  Unsupported type for extracting first element: " + value.getClass().getName());
                    }
                } else {
                    Log.w(TAG, "Output " + outputName + " not found in results.");
                }
            }
            return outputMap;

        } catch (OrtException e) {
            Log.e(TAG, "Inference failed: " + e.getMessage(), e);
            return null;
        }
    }

    /**
     * Runs inference on the ONNX model and returns the specified output.
     *
     * @param inputData  The input data for the model.
     * @param outputName The name of the output to return.
     * @return The specified output, or null if inference fails or the output is not found.
     */
    public Object runInference(float[] inputData, String outputName) {
        Log.d(TAG, "runInference() called with: inputData = [" + inputData + "], outputName = [" + outputName + "]");

        Map<String, Object> outputMap = runInference(inputData);

        if (outputMap == null) {
            return null;
        }

        if (outputMap.containsKey(outputName)) {
            return outputMap.get(outputName);
        } else {
            Log.w(TAG, "Output " + outputName + " not found in results.");
            return null;
        }
    }

    public static void printInferenceResult(Map<String, Object> result) {
        if (result == null) {
            Log.d(TAG, "Inference result is null.");
            return;
        }

        Log.d(TAG, "Inference result:");
        for (Map.Entry<String, Object> entry : result.entrySet()) {
            String key = entry.getKey();
            Object value = entry.getValue();

            Log.d(TAG, "  Key: " + key);

            if (value instanceof long[]) {
                long[] longArray = (long[]) value;
                Log.d(TAG, "    Value (long[]): " + java.util.Arrays.toString(longArray));
            } else if (value instanceof float[]) {
                float[] floatArray = (float[]) value;
                Log.d(TAG, "    Value (float[]): " + java.util.Arrays.toString(floatArray));
            } else if (value instanceof int[]) {
                int[] intArray = (int[]) value;
                Log.d(TAG, "    Value (int[]): " + java.util.Arrays.toString(intArray));
            } else if (value instanceof double[]) {
                double[] doubleArray = (double[]) value;
                Log.d(TAG, "    Value (double[]): " + java.util.Arrays.toString(doubleArray));
            } else if (value instanceof byte[]) {
                byte[] byteArray = (byte[]) value;
                Log.d(TAG, "    Value (byte[]): " + java.util.Arrays.toString(byteArray));
            } else if (value instanceof short[]) {
                short[] shortArray = (short[]) value;
                Log.d(TAG, "    Value (short[]): " + java.util.Arrays.toString(shortArray));
            } else if (value instanceof boolean[]) {
                boolean[] booleanArray = (boolean[]) value;
                Log.d(TAG, "    Value (boolean[]): " + java.util.Arrays.toString(booleanArray));
            } else if (value instanceof String[]) {
                String[] stringArray = (String[]) value;
                Log.d(TAG, "    Value (String[]): " + java.util.Arrays.toString(stringArray));
            } else if (value instanceof String) {
                Log.d(TAG, "    Value (String): " + value);
            } else if (value instanceof Long) {
                Log.d(TAG, "    Value (Long): " + value);
            } else if (value instanceof Float) {
                Log.d(TAG, "    Value (Float): " + value);
            } else if (value instanceof Integer) {
                Log.d(TAG, "    Value (Integer): " + value);
            } else if (value instanceof Double) {
                Log.d(TAG, "    Value (Double): " + value);
            } else if (value instanceof Byte) {
                Log.d(TAG, "    Value (Byte): " + value);
            } else if (value instanceof Short) {
                Log.d(TAG, "    Value (Short): " + value);
            } else if (value instanceof Boolean) {
                Log.d(TAG, "    Value (Boolean): " + value);
            } else {
                Log.d(TAG, "    Value (Unknown Type): " + value.toString());
            }
        }
    }
}