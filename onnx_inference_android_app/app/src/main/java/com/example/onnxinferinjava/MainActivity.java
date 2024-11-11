package com.example.onnxinferinjava;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import java.io.InputStream;
import java.util.Arrays;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        final EditText inputSepalLength = (EditText) this.findViewById(R.id.input_sepal_length);
        final EditText inputSepalWidth = (EditText) this.findViewById(R.id.input_sepal_width);
        final EditText inputPetalLength = (EditText) this.findViewById(R.id.input_petal_length);
        final EditText inputPetalWidth = (EditText) this.findViewById(R.id.input_petal_width);
        final TextView outputTextView = (TextView) this.findViewById(R.id.output_textview);
        Button button = (Button) this.findViewById(R.id.predict_button);
        button.setOnClickListener((View.OnClickListener) (new View.OnClickListener() {
            public final void onClick(View it) {
                float inputSepalLengthFloat = Float.parseFloat(inputSepalLength.getText().toString());
                float inputSepalWidthFloat = Float.parseFloat(inputSepalWidth.getText().toString());
                float inputPetalLengthFloat = Float.parseFloat(inputPetalLength.getText().toString());
                float inputPetalWidthFloat = Float.parseFloat(inputPetalWidth.getText().toString());
                if (true) {
                    float[] inputData = {inputSepalLengthFloat, inputSepalWidthFloat, inputPetalLengthFloat, inputPetalWidthFloat};

                    int outputClass = predictIris(inputData);
                    String outputString = "";

                    switch (outputClass) {
                        case 0:
                            outputString = "Iris Setosa";
                            break;
                        case 1:
                            outputString = "Iris Setosa";
                            break;
                        case 2:
                            outputString = "Iris Virginica";
                            break;
                        default:
                            break;
                    }
                    outputTextView.setText("Prediction: " + outputString);


                } else {
                    Toast.makeText(MainActivity.this, "Please check the inputs", Toast.LENGTH_LONG).show();
                }

            }
        }));

    }

    private int predictIris(float[] inputData) {
        InputStream modelInputStream = getResources().openRawResource(R.raw.xgbc_iris);
        ModelRunner modelRunner = null;
        try {
            modelRunner = new ModelRunner(modelInputStream, inputData.length);
//            float[] inputData = {5.7f, 3.8f, 1.7f, 0.3f};
//            float[] inputData = {6.1f, 2.8f, 4.7f, 1.2f};
//            float[] inputData = {7.7f, 2.6f, 6.9f, 2.3f};

            return (int) modelRunner.runInference(inputData);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}