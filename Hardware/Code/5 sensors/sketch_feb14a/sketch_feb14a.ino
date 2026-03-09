// ============================================================
// ESP32 + 4x JSN-SR04T (UNDERWATER) — ONE-BY-ONE TEST (SEQUENTIAL)
// With: baseline stability gate + sensor health warnings
//
// Order in arrays:
//   idx0 -> Sensor 1 (Center) : TRIG 5  ECHO 19
//   idx1 -> Sensor 2 (down)   : TRIG 17 ECHO 21
//   idx2 -> Sensor 3 (right)   : TRIG 16 ECHO 22
//   idx3 -> Sensor 4 (left)  : TRIG 25 ECHO 18
//
// Commands:
//   '0' auto cycle
//   '1','2','3','5' manual lock
//
// CSV:
// time_ms,sensor_id,echo_us,valid,dist_cm,dist_f_cm,baseline_cm,enter_thr_cm,exit_thr_cm,danger,event
// ============================================================

#include <Arduino.h>

const int N_SENS = 4;

int SENSOR_ID[N_SENS] = { 1, 2, 3, 5 };
int TRIG_PIN[N_SENS]  = { 5, 17, 16, 25 };
int ECHO_PIN[N_SENS]  = { 19, 21, 22, 18 };

// ---------- Timing ----------
const int   TRIG_HIGH_US  = 25;
const long  TIMEOUT_US    = 60000;

const int   NPINGS        = 7;
const int   INTERPING_MS  = 25;

const int   READ_PERIOD_MS  = 200;
const int   AUTO_SWITCH_MS  = 7000;   // slower cycling so baseline can lock

// ---------- Echo sanity ----------
const long  ECHO_MIN_US   = 250;
const long  ECHO_MAX_US   = 30000;

// Underwater speed
const float CM_PER_US_ONEWAY = 0.148f;

// Clamp
const float MIN_DIST_CM = 20.0f;
const float MAX_DIST_CM = 600.0f;

// Filter
const float DIST_EMA_A       = 0.25f;
const float DROP_REJECT_FRAC = 0.45f;

// ---------- Baseline ----------
const int   BASELINE_SAMPLES   = 25;     // number of accepted stable samples
const float BASELINE_STABLE_BAND_CM = 12.0f; // must stay within +/- band
const int   BASELINE_STABLE_NEED = 12;   // need this many consecutive stable reads

// danger threshold relative to baseline
const float DANGER_MARGIN_CM = 60.0f;    // enter_thr = baseline - margin
const float HYST_CM          = 20.0f;

// confirmation window
const int CONF_N  = 3;
const int ENTER_K = 2;
const int EXIT_K  = 2;

// sensor health
const int NOECHO_WARN_N = 8;  // after N consecutive invalid reads show warning

// ============================================================

struct SensorState {
  bool  dist_init = false;
  float dist_f_cm = 0.0f;

  // baseline
  bool  baseline_ready = false;
  int   base_n = 0;
  float base_buf[BASELINE_SAMPLES];
  float baseline_cm = 0.0f;

  // baseline stability gate
  float stable_ref = 0.0f;
  int   stable_count = 0;

  // danger
  bool  in_danger = false;
  float hist[CONF_N];
  int   hist_i = 0;
  int   hist_count = 0;

  // health
  int   noEchoCount = 0;
};

SensorState ST[N_SENS];

int modeIndex = -1; // -1 auto, else fixed index
int activeIdx = 0;

unsigned long lastSwitchMs = 0;
unsigned long lastReadMs = 0;

// ============================================================
// Helpers
float clampf(float x, float lo, float hi) { return (x < lo) ? lo : (x > hi) ? hi : x; }

void sortArray(float *a, int n) {
  for (int i = 0; i < n - 1; i++)
    for (int j = i + 1; j < n; j++)
      if (a[j] < a[i]) { float t = a[i]; a[i] = a[j]; a[j] = t; }
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

  for (int i = 0; i < k - 1; i++)
    for (int j = i + 1; j < k; j++)
      if (vals[j] < vals[i]) { long t = vals[i]; vals[i] = vals[j]; vals[j] = t; }

  return vals[k / 2];
}

float echoToDistCm(long echo_us) { return (echo_us * CM_PER_US_ONEWAY) / 2.0f; }

float dangerScore(float dist_f, float enter_thr) {
  const float WIDTH = 120.0f;
  if (dist_f <= enter_thr) return 1.0f;
  if (dist_f >= enter_thr + WIDTH) return 0.0f;
  return 1.0f - (dist_f - enter_thr) / WIDTH;
}

int sensorIdToIndex(int sid) {
  for (int i = 0; i < N_SENS; i++) if (SENSOR_ID[i] == sid) return i;
  return -1;
}

void pollSerialMode() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '0') {
      modeIndex = -1;
      activeIdx = 0;
      lastSwitchMs = millis();
      Serial.println("MODE: AUTO cycle 1->2->3->5");
    } else if (c == '1' || c == '2' || c == '3' || c == '5') {
      int sid = c - '0';
      int idx = sensorIdToIndex(sid);
      if (idx >= 0) {
        modeIndex = idx;
        Serial.print("MODE: MANUAL lock sensor ");
        Serial.println(sid);
      }
    }
  }
}

// ============================================================
// Baseline stability-gated accept
bool baselineAccept(SensorState &st, float x) {
  if (st.stable_count == 0) {
    st.stable_ref = x;
    st.stable_count = 1;
    return false;
  }

  // if within band -> stable++
  if (fabsf(x - st.stable_ref) <= BASELINE_STABLE_BAND_CM) {
    st.stable_ref = 0.8f * st.stable_ref + 0.2f * x;
    st.stable_count++;
  } else {
    // reset stability if it jumps
    st.stable_ref = x;
    st.stable_count = 1;
  }

  // only accept samples after enough stable reads
  return (st.stable_count >= BASELINE_STABLE_NEED);
}

// ============================================================
// One sensor read + print
void readAndPrint(int idx) {
  int sid = SENSOR_ID[idx];
  int trigPin = TRIG_PIN[idx];
  int echoPin = ECHO_PIN[idx];
  SensorState &st = ST[idx];

  long echo_us = readEchoMedian(trigPin, echoPin);
  bool valid = (echo_us != 0);

  float dist_cm = 0.0f;
  float danger = 0.0f;
  int   event = 0;

  float enter_thr = 0.0f;
  float exit_thr  = 0.0f;

  if (!valid) {
    st.noEchoCount++;
    if (st.noEchoCount == NOECHO_WARN_N) {
      Serial.print("WARNING: sensor ");
      Serial.print(sid);
      Serial.println(" no echo (check wiring / level shift / pin)!");
    }
  } else {
    st.noEchoCount = 0;
  }

  if (valid) {
    dist_cm = clampf(echoToDistCm(echo_us), MIN_DIST_CM, MAX_DIST_CM);

    if (!st.dist_init) {
      st.dist_f_cm = dist_cm;
      st.dist_init = true;
    } else {
      if (dist_cm < st.dist_f_cm * (1.0f - DROP_REJECT_FRAC)) {
        // reject glitch
      } else {
        st.dist_f_cm = DIST_EMA_A * dist_cm + (1.0f - DIST_EMA_A) * st.dist_f_cm;
      }
    }

    // Baseline learning
    if (!st.baseline_ready) {
      bool ok = baselineAccept(st, st.dist_f_cm);
      if (ok) {
        st.base_buf[st.base_n++] = st.dist_f_cm;
      }

      if (st.base_n >= BASELINE_SAMPLES) {
        sortArray(st.base_buf, BASELINE_SAMPLES);
        st.baseline_cm = st.base_buf[BASELINE_SAMPLES / 2];
        st.baseline_ready = true;

        Serial.print("BASELINE learned for sensor ");
        Serial.print(sid);
        Serial.print(": ");
        Serial.print(st.baseline_cm, 1);
        Serial.println(" cm");
      }

      danger = 0.0f;
      event = 0;
    } else {
      enter_thr = clampf(st.baseline_cm - DANGER_MARGIN_CM, MIN_DIST_CM, MAX_DIST_CM);
      exit_thr  = clampf(enter_thr + HYST_CM, MIN_DIST_CM, MAX_DIST_CM);

      pushHist(st, st.dist_f_cm);

      if (!st.in_danger) {
        if (st.hist_count == CONF_N && countBelow(st, enter_thr) >= ENTER_K) {
          st.in_danger = true;
          event = +1;
        }
      } else {
        if (st.hist_count == CONF_N && countAbove(st, exit_thr) >= EXIT_K) {
          st.in_danger = false;
          event = -1;
        }
      }

      danger = st.in_danger ? dangerScore(st.dist_f_cm, enter_thr) : 0.0f;
    }
  } else {
    // invalid echo: keep state, don't force danger
    if (st.baseline_ready) {
      enter_thr = clampf(st.baseline_cm - DANGER_MARGIN_CM, MIN_DIST_CM, MAX_DIST_CM);
      exit_thr  = clampf(enter_thr + HYST_CM, MIN_DIST_CM, MAX_DIST_CM);
    }
    danger = 0.0f;
    event = 0;
  }

  // Print CSV
  Serial.print(millis()); Serial.print(",");
  Serial.print(sid); Serial.print(",");
  Serial.print(echo_us); Serial.print(",");
  Serial.print(valid ? 1 : 0); Serial.print(",");
  Serial.print(valid ? dist_cm : 0.0f, 1); Serial.print(",");
  Serial.print(st.dist_f_cm, 1); Serial.print(",");
  Serial.print(st.baseline_ready ? st.baseline_cm : 0.0f, 1); Serial.print(",");
  Serial.print(st.baseline_ready ? enter_thr : 0.0f, 1); Serial.print(",");
  Serial.print(st.baseline_ready ? exit_thr  : 0.0f, 1); Serial.print(",");
  Serial.print(danger, 3); Serial.print(",");
  Serial.println(event);
}

// ============================================================

void setup() {
  Serial.begin(115200);
  delay(300);

  for (int i = 0; i < N_SENS; i++) {
    pinMode(TRIG_PIN[i], OUTPUT);
    digitalWrite(TRIG_PIN[i], LOW);
    pinMode(ECHO_PIN[i], INPUT);   // MUST be level shifted to 3.3V
  }

  Serial.println("ESP32 4-SENSOR TEST (active sensors 1,2,3,5)");
  Serial.println("CSV: time_ms,sensor_id,echo_us,valid,dist_cm,dist_f_cm,baseline_cm,enter_thr_cm,exit_thr_cm,danger,event");
  Serial.println("Commands: '0'=auto, '1','2','3','5'=manual lock");
  Serial.println("Baseline only learns when readings are STABLE (open water).");

  activeIdx = 0;
  modeIndex = -1;
  lastSwitchMs = millis();
  lastReadMs = millis();
}

void loop() {
  pollSerialMode();
  unsigned long now = millis();

  int idx = activeIdx;

  if (modeIndex == -1) {
    if (now - lastSwitchMs >= (unsigned long)AUTO_SWITCH_MS) {
      activeIdx = (activeIdx + 1) % N_SENS;
      lastSwitchMs = now;

      Serial.print("---- NOW TESTING SENSOR ");
      Serial.print(SENSOR_ID[activeIdx]);
      Serial.println(" ----");
    }
    idx = activeIdx;
  } else {
    idx = modeIndex;
  }

  if (now - lastReadMs >= (unsigned long)READ_PERIOD_MS) {
    lastReadMs = now;
    readAndPrint(idx);
  }
}

