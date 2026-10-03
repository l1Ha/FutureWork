package io.github.l1ha.futurework;

import android.content.Context;
import android.os.Build;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.webkit.JavascriptInterface;
import android.widget.Toast;

public class WebAppInterface {
    private final MainActivity mActivity;
    private final Vibrator mVibrator;

    public WebAppInterface(MainActivity activity) {
        this.mActivity = activity;
        this.mVibrator = (Vibrator) activity.getSystemService(Context.VIBRATOR_SERVICE);
    }

    @JavascriptInterface
    public void startVoiceInput() {
        mActivity.runOnUiThread(() -> mActivity.launchSpeechRecognizer());
    }

    @JavascriptInterface
    public void vibrate(long milliseconds) {
        if (mVibrator != null && mVibrator.hasVibrator()) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                mVibrator.vibrate(VibrationEffect.createOneShot(milliseconds, VibrationEffect.DEFAULT_AMPLITUDE));
            } else {
                mVibrator.vibrate(milliseconds);
            }
        }
    }

    @JavascriptInterface
    public void showToast(String message) {
        mActivity.runOnUiThread(() -> Toast.makeText(mActivity, message, Toast.LENGTH_SHORT).show());
    }

    @JavascriptInterface
    public String getPlatformInfo() {
        return "Android " + Build.VERSION.RELEASE + " (API " + Build.VERSION.SDK_INT + ") - " + Build.MODEL;
    }
}
