package com.jascal.stylized;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Handler;
import android.os.SystemClock;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends AppCompatActivity {
    private static final String MODEL_FILE = "stylize_quantized.pb";
    private static final String INPUT_NODE = "input";
    private static final String STYLE_NODE = "style_num";
    private static final int NUM_STYLES = 26;
    private static final String OUTPUT_NODE = "transformer/expand/conv3/conv/Sigmoid";

    private TensorFlowInferenceInterface inferenceInterface;
    private Handler handler;

    private long lastProcessingTimeMs;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;
    private Bitmap textureCopyBitmap;

    private ImageView imageView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.origin);
        croppedBitmap = BitmapFactory.decodeResource(getResources(), R.mipmap.style0);
        croppedBitmap = Bitmap.createScaledBitmap(croppedBitmap, desiredSize, desiredSize, false);
        imageView.setImageBitmap(croppedBitmap);
    }

    public void go(View view){
        execute();
    }

    private void execute() {
        handler = new Handler(getMainLooper());
        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);
        runInBackground(new Runnable() {
            @Override
            public void run() {
                cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                final long startTime = SystemClock.uptimeMillis();
                stylizeImage(croppedBitmap);
                lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                textureCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                done();
            }
        });
    }

    private void done() {
        imageView.setImageBitmap(textureCopyBitmap);
    }

    private int desiredSize = 256;
    private final float[] styleVals = new float[NUM_STYLES];
    private int[] intValues = new int[desiredSize * desiredSize];
    private float[] floatValues = new float[desiredSize * desiredSize * 3];
    private void stylizeImage(Bitmap bitmap) {
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3] = ((val >> 16) & 0xFF) / 255.0f;
            floatValues[i * 3 + 1] = ((val >> 8) & 0xFF) / 255.0f;
            floatValues[i * 3 + 2] = (val & 0xFF) / 255.0f;
        }

        float style;
        for (int i = 0; i < NUM_STYLES; ++i) {
            style = (float) Math.random();

            styleVals[i] = style / NUM_STYLES;
        }

        // Copy the input data into TensorFlow.
        Log.d("tensor", "Width: " + bitmap.getWidth() + ", Height: " + bitmap.getHeight());
        inferenceInterface.feed(
                INPUT_NODE, floatValues, 1, bitmap.getWidth(), bitmap.getHeight(), 3);
        inferenceInterface.feed(STYLE_NODE, styleVals, NUM_STYLES);

        inferenceInterface.run(new String[]{OUTPUT_NODE}, false);
        inferenceInterface.fetch(OUTPUT_NODE, floatValues);

        for (int i = 0; i < intValues.length; ++i) {
            intValues[i] =
                    0xFF000000
                            | (((int) (floatValues[i * 3] * 255)) << 16)
                            | (((int) (floatValues[i * 3 + 1] * 255)) << 8)
                            | ((int) (floatValues[i * 3 + 2] * 255));
        }

        bitmap.setPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    }

    protected synchronized void runInBackground(final Runnable r) {
        if (handler != null) {
            handler.post(r);
        }
    }
}
