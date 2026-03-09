// ============================================================
// ESP32 + 4x JSN-SR04T (UNDERWATER) — FULL SCAN FOR JETSON/SNN
// Sensors:
//   0 CENTER: TRIG 5  ECHO 19
//   1 DOWN  : TRIG 17 ECHO 21
//   2 RIGHT : TRIG 16 ECHO 22
//   3 LEFT  : TRIG 25 ECHO 18
//
// CSV (one line per full scan):
// t_ms, dC,dD,dR,dL, sC,sD,sR,sL, eC,eD,eR,eL
// ============================================================

#include <Arduino.h>
#include <math.h>

#define N_SENS 4

// Order: Center, Down, Right, Left
int TRIG_PIN[N_SENS] = { 5, 17, 16, 25 };
int ECHO_PIN[N_SENS] = { 19, 21, 22, 18 };

// ---------- Timing ----------
const int   TRIG_HIGH_US     = 25;
const long  TIMEOUT_US       = 20000;   // reduce long multipath (20ms)
const int   SCAN_PERIOD_MS   = 250;     // ~4 Hz full scan
const int   BETWEEN_SENS_MS  = 35;      // quiet time between sensors

// Median-of-N (keep light for 4 sensors)
const int   NPINGS       = 3;
const int   INTERPING_MS = 12;

// ---------- Echo sanity ----------
const long  ECHO_MIN_US = 250;
const long  ECHO_MAX_US = 30000;

// Underwater speed approx
const float CM_PER_US_ONEWAY = 0.148f;

// Clamp
const float MIN_DIST_CM = 20.0f;
const float MAX_DIST_CM = 600.0f;

// Filter
const float DIST_EMA_A       = 0.25f;
const float DROP_REJECT_FRAC = 0.35f;
const int   DROP_STRIKE_NEED = 2;        // 2-strike: accept real drops

// ---------- Baseline ----------
const int   BASELINE_SAMPLES = 25;
const float BASELINE_STABLE_BAND_CM = 12.0f;
const int   BASELINE_STABLE_NEED    = 10;

// Baseline tracking when safe (pool robustness)
const float BASELINE_TRACK_A   = 0.02f;
const float BASELINE_MAX_STEP  = 6.0f;

// Danger thresholds relative to baseline
const float DANGER_MARGIN_CM = 60.0f;
const float HYST_CM          = 20.0f;

// confirmation window
const int CONF_N  = 5;
const int ENTER_K = 4;
const int EXIT_K  = 4;

// ============================================================

float clampf(float x, float lo, float hi) { return (x < lo) ? lo : (x > hi) ? hi : x; }

void sortLong(long *a, int n) {
  for (int i = 0; i < n - 1; i++)
    for (int j = i + 1; j < n; j++)
      if (a[j] < a[i]) { long t = a[i]; a[i] = a[j]; a[j] = t; }
}

void sortFloat(float *a, int n) {
  for (int i = 0; i < n - 1; i++)
    for (int j = i + 1; j < n; j++)
      if (a[j] < a[i]) { float t = a[i]; a[i] = a[j]; a[j] = t; }
}

long readEchoOnce(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(5);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(TRIG_HIGH_US);
  digitalWrite(trigPin, LOW);
  return pulseIn(echoPin, HIGH, TIMEOUT_US);
}

long readEchoMedian(int trigPin, int echoPin) {
  long vals[NPINGS];
  int k = 0;
  for (int i = 0; i < NPINGS; i++) {
    long e = readEchoOnce(trigPin, echoPin);
    if (e >= ECHO_MIN_US && e <= ECHO_MAX_US) vals[k++] = e;
    delay(INTERPING_MS);
  }
  if (k == 0) return 0;
  sortLong(vals, k);
  return vals[k / 2];
}

float echoToDistCm(long echo_us) {
  return (echo_us * CM_PER_US_ONEWAY) / 2.0f;
}

float dangerScore(float dist_f, float enter_thr) {
  const float WIDTH = 120.0f;
  if (dist_f <= enter_thr) return 1.0f;
  if (dist_f >= enter_thr + WIDTH) return 0.0f;
  return 1.0f - (dist_f - enter_thr) / WIDTH;
}

struct SensorState {
  bool  dist_init = false;
  float dist_f_cm = 0.0f;

  // baseline
  bool  baseline_ready = false;
  int   base_n = 0;
  float base_buf[BASELINE_SAMPLES];
  float baseline_cm = 0.0f;

  // stability gate
  float stable_ref = 0.0f;
  int   stable_count = 0;

  // danger
  bool  in_danger = false;
  float hist[CONF_N];
  int   hist_i = 0;
  int   hist_count = 0;

  // 2-strike drop
  int   drop_strikes = 0;
};

SensorState ST[N_SENS];

bool baselineAccept(SensorState &st, float x) {
  if (st.stable_count == 0) {
    st.stable_ref = x;
    st.stable_count = 1;
    return false;
  }
  if (fabsf(x - st.stable_ref) <= BASELINE_STABLE_BAND_CM) {
    st.stable_ref = 0.8f * st.stable_ref + 0.2f * x;
    st.stable_count++;
  } else {
    st.stable_ref = x;
    st.stable_count = 1;
  }
  return (st.stable_count >= BASELINE_STABLE_NEED);
}

void pushHist(SensorState &st, float x) {
  st.hist[st.hist_i] = x;
  st.hist_i = (st.hist_i + 1) % CONF_N;
  if (st.hist_count < CONF_N) st.hist_count++;
}

int countBelow(const SensorState &st, float thr) {
  int c = 0;
  for (int i = 0; i < st.hist_count; i++) if (st.hist[i] <= thr) c++;
  return c;
}

int countAbove(const SensorState &st, float thr) {
  int c = 0;
  for (int i = 0; i < st.hist_count; i++) if (st.hist[i] >= thr) c++;
  return c;
}

void setup() {
  Serial.begin(115200);
  delay(300);

  for (int i = 0; i < N_SENS; i++) {
    pinMode(TRIG_PIN[i], OUTPUT);
    digitalWrite(TRIG_PIN[i], LOW);
    pinMode(ECHO_PIN[i], INPUT);  // IMPORTANT: level shift ECHO to 3.3V
  }

  Serial.println("t_ms,dC,dD,dR,dL,sC,sD,sR,sL,eC,eD,eR,eL");
  Serial.println("# Calibrate: keep robot in FINAL pose for ~6-8s, then test obstacles.");
}

void loop() {
  unsigned long t0 = millis();

  float dangerOut[N_SENS] = {0};
  int   stateOut[N_SENS]  = {0};
  int   eventOut[N_SENS]  = {0};

  for (int i = 0; i < N_SENS; i++) {
    SensorState &st = ST[i];

    long echo_us = readEchoMedian(TRIG_PIN[i], ECHO_PIN[i]);
    bool valid = (echo_us != 0);

    eventOut[i] = 0;

    if (valid) {
      float dist_cm = clampf(echoToDistCm(echo_us), MIN_DIST_CM, MAX_DIST_CM);

      // ---- Filter with 2-strike drop handling ----
      if (!st.dist_init) {
        st.dist_f_cm = dist_cm;
        st.dist_init = true;
        st.drop_strikes = 0;
      } else {
        float reject_limit = st.dist_f_cm * (1.0f - DROP_REJECT_FRAC);
        if (dist_cm < reject_limit) {
          st.drop_strikes++;
          if (st.drop_strikes >= DROP_STRIKE_NEED) {
            st.dist_f_cm = DIST_EMA_A * dist_cm + (1.0f - DIST_EMA_A) * st.dist_f_cm;
            st.drop_strikes = DROP_STRIKE_NEED;
          }
        } else {
          st.drop_strikes = 0;
          st.dist_f_cm = DIST_EMA_A * dist_cm + (1.0f - DIST_EMA_A) * st.dist_f_cm;
        }
      }

      // ---- Baseline learning ----
      if (!st.baseline_ready) {
        bool ok = baselineAccept(st, st.dist_f_cm);
        if (ok) st.base_buf[st.base_n++] = st.dist_f_cm;

        if (st.base_n >= BASELINE_SAMPLES) {
          sortFloat(st.base_buf, BASELINE_SAMPLES);
          st.baseline_cm = st.base_buf[BASELINE_SAMPLES / 2];
          st.baseline_ready = true;

          st.in_danger = false;
          st.hist_count = 0;
          st.hist_i = 0;
          st.drop_strikes = 0;
        }

        dangerOut[i] = 0.0f;
        stateOut[i]  = 0;
        delay(BETWEEN_SENS_MS);
        continue;
      }

      // ---- baseline tracking when safe ----
      if (!st.in_danger) {
        float delta = st.dist_f_cm - st.baseline_cm;
        delta = clampf(delta, -BASELINE_MAX_STEP, BASELINE_MAX_STEP);
        st.baseline_cm += BASELINE_TRACK_A * delta;
      }

      float enter_thr = clampf(st.baseline_cm - DANGER_MARGIN_CM, MIN_DIST_CM, MAX_DIST_CM);
      float exit_thr  = clampf(enter_thr + HYST_CM, MIN_DIST_CM, MAX_DIST_CM);

      pushHist(st, st.dist_f_cm);

      if (!st.in_danger) {
        if (st.hist_count == CONF_N && countBelow(st, enter_thr) >= ENTER_K) {
          st.in_danger = true;
          eventOut[i] = +1;
        }
      } else {
        if (st.hist_count == CONF_N && countAbove(st, exit_thr) >= EXIT_K) {
          st.in_danger = false;
          eventOut[i] = -1;
        }
      }

      dangerOut[i] = st.in_danger ? dangerScore(st.dist_f_cm, enter_thr) : 0.0f;
      stateOut[i]  = st.in_danger ? 1 : 0;

    } else {
      // invalid echo -> don't force danger
      dangerOut[i] = 0.0f;
      stateOut[i]  = ST[i].in_danger ? 1 : 0;
      eventOut[i]  = 0;
    }

    delay(BETWEEN_SENS_MS);
  }

  // Print one scan line
  Serial.print(t0);
  for (int i = 0; i < N_SENS; i++) { Serial.print(","); Serial.print(dangerOut[i], 3); }
  for (int i = 0; i < N_SENS; i++) { Serial.print(","); Serial.print(stateOut[i]); }
  for (int i = 0; i < N_SENS; i++) { Serial.print(","); Serial.print(eventOut[i]); }
  Serial.println();

  // hold scan rate
  unsigned long spent = millis() - t0;
  if (spent < (unsigned long)SCAN_PERIOD_MS) delay(SCAN_PERIOD_MS - spent);
}


