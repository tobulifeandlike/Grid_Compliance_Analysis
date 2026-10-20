import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import find_peaks

class Analyzer:
  # initialization
  def __init__(self, v_low=0.90, v_high=1.05, loading_limit_pct=100.0, fn=60.0, pfr_db_hz=0.0, icr_pu=1.0):
    self.v_low = v_low                      # lower edge of the continuous operation region (pu)
    self.v_high = v_high                    # upper edge of the continuous operation region (pu), the 1.05 setup only applies to system with voltage lower than 500 kV
    self.loading_limit_pct = loading_limit_pct
    self.fn = fn                            # rated frequency (Hz)
    self.pfr_db_hz = pfr_db_hz              # plant PFR deadband (Hz); IEEE 2800 default is 0.06% * fn
    self.icr_pu = icr_pu   # ICR in pu of the apparent-power base S_base (ICR_MW / S_base_MVA)
    # Measurement levels, IEEE 2800-2022 Cl.7.2.2.1 / Cl.7.3.2 (limits live in the comparator)
    # (direction, level, sliding window in s, tag)
    self.lvrt_bands = [("below", 0.90, 10.0, "V<0.90"), ("below", 0.70, 10.0, "V<0.70"),
                     ("below", 0.50, 10.0, "V<0.50"), ("below", 0.25, 10.0, "V<0.25")]
    self.hvrt_bands = [("above", 1.10, 10.0, "V>1.10"),("above", 1.05, 3600.0, "V>1.05")]
    self.freq_bands = [("below", 58.8, 600.0, "f<58.8"), ("above", 61.2, 600.0, "f>61.2")]

  '''
  Key parameters
  '''
  def _max_rocof(self, f, t, T = 0.1):
    '''
    Maximum rate of change of frequency absolute value, based on IEEE 2800-2022 Cl7.3 |RoCoF| ≤ 5.0 Hz/s, NERC PRC-029-1
    '''
    # Frequency at each window start; NaN if before the record
    # Array-based operation
    f_start = np.interp(t-T, t, f, left = np.nan)
    rocof = (f - f_start)/T
    # Find index of the max |rocof|, skipping the NaN value
    i_max = np.nanargmax(np.abs(rocof))
    return float(abs(rocof[i_max])), float(t[i_max])

  def _effective_inertia(self, f, p, t, t0, t1):
    '''
    H = -fn * ΔE/(2.Δf) from the rotor equation, 2H/fn·d(Δf)/dt = -ΔP
    where ΔE is the change in kinetic energy, Δf is the change in frequency, fn is the rated frequency, p(t) is the power injection, and p0 is the initial power injection.
    '''
    # Acquire at least 0.2s time window to get the steady state data
    m_pre = (t >= t0 - 0.2) & (t < t0)
    p_pre = float(np.mean(p[m_pre]))        # Average power before the disturbance
    # Time window for intgration
    m_win = (t >= t0) & (t <= t1)
    # Inject increment energy to the window: ΔE = ∫(P − P_pre)dt（trapezoidal method）
    dE = float(trapezoid(p[m_win] - p_pre, t[m_win]))
    # Frequency line fitting, |Δf| = |slope|×(t1−t0)
    df = abs(np.polyfit(t[m_win], f[m_win], 1)[0]) * (t1 - t0)
    if df < 1e-6:                           # Zero frequency change, cannot calculate inertia
        return None
    return -self.fn * dE / (2.0 * df)

  def _phase_jump_dp(self, p,t,t_jump):
    '''
    Calculate the active power change at the 1/3 cycle after phase jump 
    DOE UNIFI GFM spec v3 §2.2.5, GFM inject/absorb power to counteract the phase angle change
    '''
    cyc = 1.0 / self.fn                     # 1 cycle count（s）
    p_pre = float(np.mean(p[(t >= t_jump - 0.1) & (t < t_jump)]))

    dP1 = float(np.interp(t_jump + 1.0*cyc, t, p)) - p_pre
    dP3 = float(np.interp(t_jump + 3.0*cyc, t, p)) - p_pre
    return dP1, dP3 


  def _pfr_droop(self, f, p, t, t_pre0, t_pre1, t_post0, t_post1, p_rated):
    '''
    Quasi steady state droop calculation based on FERC Order 842 - all newly built generation units should be less than or equal to 5%, IEEE 2800
    '''
    f_pre = float(np.mean(f[(t >= t_pre0) & (t <= t_pre1)]))
    f_post = float(np.mean(f[(t >= t_post0) & (t <= t_post1)]))
    p_pre = float(np.mean(p[(t >= t_pre0) & (t <= t_pre1)]))
    p_post = float(np.mean(p[(t >= t_post0) & (t <= t_post1)]))
    # droop% = (Δf/fn) / ΔP × 100, Δf = f_pre − f_post, ΔP = (P_post − P_pre) / P_rated
    dp = (p_post - p_pre) / p_rated
    if abs(dp) < 1e-9:
      return None
    df = f_pre - f_post
    # Compute an effective frequency deviation based on the 6.1.1 Table 6 equation with dead zone 
    df_eff = np.sign(df) * max(abs(df)-self.pfr_db_hz, 0.0)
    return (df_eff/ self.fn) / dp * 100.0 # Regulate picking positive droop value

  def analyze_key_parameters(self, f, t, p, ev):
    """Assemble key parameters.
      Args:
      f, t, p: frequency / time / active-power arrays (already float).
      ev (dict): event contract; keys are fixed:
        freq_t0 (float)      - start time of the frequency event
        inertia_t0 (float)   - start of the inertia evaluation window
        inertia_t1 (float)   - end of the inertia evaluation window
        phase_jump_t (float) - time of the phase jump
        pfr_pre (tuple)      - pre-disturbance quasi-steady window (t0, t1)
        pfr_post (tuple)     - post-disturbance quasi-steady window (t0, t1)
        p_rated (float)      - ICR, the continuous ACTIVE-power rating (IEEE 2800 Cl.3), in the unit of p.p is normally in pu of the apparent-power base S_base (MVA), so p_rated = ICR_MW / S_base_MVA (e.g. 90 MW / 100 MVA = 0.9). 
    """
    f = np.array(f, dtype=float)
    t = np.array(t, dtype=float)
    p = np.array(p, dtype=float)
    m = t >= ev['freq_t0']
    seg, tseg = f[m], t[m]
    i_min, i_max = int(np.argmin(seg)), int(np.argmax(seg))
    # nadir: minimum value of the frequency during the frequency event
    nadir, t_nadir = float(seg[i_min]), float(tseg[i_min])
    # zenith: maximum value of the frequency during the frequency event
    zenith = float(seg[i_max])

    rocof_max, _ = self._max_rocof(f, t)
    H = self._effective_inertia(f, p, t, ev["inertia_t0"], ev["inertia_t1"])
    dP1, dP3 = self._phase_jump_dp(p, t, ev["phase_jump_t"])
    pre0, pre1 = ev["pfr_pre"]
    post0, post1 = ev["pfr_post"]
    droop_percent = self._pfr_droop(f, p, t, pre0, pre1, post0, post1, ev["p_rated"])


    out = {
      "freq_nadir_Hz": {"value": nadir, "unit": "Hz", "desc_zh": "Minimum frequency after disturbance"},
      "freq_zenith_Hz": {"value": zenith, "unit": "Hz", "desc_zh": "Maximum frequency after disturbance"},
      "rocof_ride_through_Hz_per_s": {"value": rocof_max, "unit": "Hz/s", "desc_zh": "Max |RoCoF|"},
      "unifi_equiv_inertia_H_s": {"value": H, "unit": "s", "desc_zh": "Equivalent inertia constant"},
      "unifi_phase_jump_test_active_power_pu": {"value": dP1, "unit": "pu", "desc_zh": "Active power variation at first cycle after the phase angle jump"},
    "unifi_phase_jump_test_active_power_pu3": {"value": dP3, "unit": "pu", "desc_zh": "Active power variation at third cycle after the phase angle jump"},
      "pfr_droop_percent": {"value": droop_percent, "unit": "%", "desc_zh": "Primary frequency modulation droop coefficient"},
    }
    return out

  '''
  Stability Analysis 
  '''
  def _load_volatility(self, p_load, t):
    """
    Load fluctuation intensity: max |dP/dt|. Ramp rate
    NERC Level 3 warning —— GW level computation load operation
    """
    # Take max absolute value and corresponding time after numerical differentiation
    raw = np.gradient(p_load, t)
    i_max = np.argmax(np.abs(raw))
    return float(abs(raw[i_max])), float(t[i_max])

  def _damping_log_decrement(self, x, t):
    """
    Log decaying method for damping ratio estimation
    NERC oscillation analysis —— ζ<5% investigate, ζ<3% modify (pending, warn only);
    δ = mean(ln(A_n/A_{n+1})), A_n is magnitude of positive peak
    ζ = δ/√(4π²+δ²)；ζ<0 indicates increasing oscillation
    """
    x = np.asarray(x, dtype = float)
    t = np.asarray(t, dtype = float)
    # 1. Estimate the noise level
    noise = np.std(np.diff(x))/np.sqrt(2) # sample-to-sample jitter, not affected by the slow oscillation
    # 2. 20 ms moving average: removes noise, keeps oscillations below ~5 Hz
    w = 0.02 
    area = np.concatenate(([0.0], np.cumsum(0.5 * (x[1:] + x[:-1]) * np.diff(t))))   # integral of x
    x_s = (np.interp(t + w / 2, t, area) - np.interp(t - w / 2, t, area)) / w        # mean of x over [t-w/2, t+w/2]
    n_avg = max(w / np.median(np.diff(t)), 1.0) # samples averaged in one window
    noise = noise / np.sqrt(n_avg) # noise left after averaging
    # 3. Settling value
    x_final = np.mean(x[t >= t[-1] - 1.0])  # settled value: mean of the last 1 s
    # 4. Remove settled value
    y = x_s - x_final  # oscillation around the settled value
    # 5. Find peaks
    thr = max(5 * noise, 1e-9) # A real peak stands 5 sigma above the noise, and emphasize the 5 sigma threshold locally (prominence)
    idx, _ = find_peaks(y, height=thr, prominence=thr)
    inside = (t[idx] > t[0] + w / 2) & (t[idx] < t[-1] - w / 2)   # the average is incomplete at the edges, remove the peaks at beginning and end 
    # 6. Find the first six peaks since the peak amplitude is high with greater information to noise ratio
    idx = idx[inside][:6]                   # at most 6 peaks
    # 7. Choose the equation
    if len(idx) >= 2:                       # log decrement over n periods
        n = len(idx) - 1
        delta = np.log(y[idx[0]] / y[idx[-1]]) / n
        zeta = delta / np.sqrt(4 * np.pi**2 + delta**2)
        period = float(np.mean(np.diff(t[idx])))
        return float(zeta), period
    if len(idx) == 1:                       # well damped: only one overshoot is visible
        step = x_final - x[0]                 # size of the step response
        O = y[idx[0]] / abs(step)             # per-unit overshoot
        zeta = -np.log(O) / np.sqrt(np.pi**2 + np.log(O)**2)
        return float(zeta), None
    return None, None                       # no oscillation found

  def analyze_stability(self, x, t):
    """
    Assemble stability measures

    output: voltage_control_damping_ratio (ζ), damping_period_s (s)。
    """
    zeta, period = self._damping_log_decrement(x,t)
    out = {
      "voltage_control_damping_ratio": {"value": zeta, "unit": "", "desc_zh": "Damping ratio (log decrement)"},
      "damping_period_s": {"value": period, "unit": "s", "desc_zh": "Dominant oscillation period"},
    }
    return out 

  '''
  Ride-through Analysis
  '''
  def _band_dwell(self, x, t,bands):
    """
    Standard: IEEE 2800-2022 Cl.7.2.2.3   (LVRT) / Cl.7.2.2.2 (HVRT) / Cl.7.3 (frequency).
    Each band accumulates dwell time separately; pass iff dwell <= limit per band.
    (Overlaying the trajectory on the envelope geometrically is the WRONG method.)
    No open-source implementation exists; hand-written with numpy masks.)
    """
    x = np.asarray(x, dtype=float)  # float array; no copy if it already is one
    t = np.asarray(t, dtype=float)
    dt = np.diff(t, append=t[-1])           # time each sample stands for: interval to the next sample
    results = []
    for direction, level, window, tag in bands:
        if direction == "below":
            hit = x < level                     # samples below the level (nested: includes deeper samples)
        else:
            hit = x > level                     # samples above the level
        c = np.cumsum(dt * hit)               # running cumulative time beyond the level from the beginning
        # Look up the computed cumulated time in the previous step
        # t - window indicates the time to look up
        # t, c: known discrete curve
        # c_start: c(t-window)
        # Obtain the time voltage violate the threshold in each 10s side window  
        c_start = np.interp(t - window, t, c, left=0.0)   # cumulative time at each window start (0 before the record)
        results.append({"tag": tag, "dwell_s": float(np.max(c - c_start))})   # worst window
    return results

  def _detect_fault_window(self, v,t):
    v = np.asarray(v, dtype=float)  # float array; no copy if it already is one
    t = np.asarray(t, dtype=float)
    idx = np.where((v < self.v_low) | (v > self.v_high))[0]   # outside the continuous operation region (Cl.7.2.2.4), tuple structure
    '''
    Example code
    v = np.array([1.00, 0.85, 0.60, 0.95, 1.08])
    mask = (v < 0.90) | (v > 1.05)    # array([False,  True,  True, False,  True])

    np.where(mask)       # (array([1, 2, 4]),)   ← tuple, note the trailing comma
    np.where(mask)[0]    #  array([1, 2, 4])     ← the index array itself
    '''
    if len(idx) == 0:
        t_start, t_end = None, None           # no fault detected
    else:
        t_start, t_end = float(t[idx[0]]), float(t[idx[-1]])
    return t_start, t_end

  def _recovery_time(self, v, p, t, t_start, t_end):
    """
    Standard: IEEE 2800-2022 Cl.7.2.2.6  
    """
    v = np.asarray(v, dtype=float)  # float array; no copy if it already is one
    p = np.asarray(p, dtype=float)
    t = np.asarray(t, dtype=float)
    m_pre = (t >= t_start - 0.5) & (t < t_start)
    if not np.any(m_pre):
        return None, None, None            # data before the fault lacks
    p_pre = float(np.mean(p[m_pre]))       # power before the disturbance
    arr = t[(t > t_end) & (v >= self.v_low) & (v <= self.v_high)]   # voltage back in the continuous region
    if len(arr) == 0:                       # voltage never recovers
        return None, p_pre, None
    t_vrec = float(arr[0])     
    p_dip = float(np.min(p[(t >= t_start) & (t <= t_vrec)]))      # lowest power during the disturbance
    band = max(0.025 * (p_pre - p_dip), 0.005*self.icr_pu)        # Table 8 settling band (p in pu of ICR)
    arr2 = t[(t >= t_vrec) & (p >= p_pre - band)]                  # power back inside the band
    if len(arr2) == 0:                      # active power never recovers
        return None, p_pre, p_dip
    return float(arr2[0]) - t_vrec, p_pre, p_dip

  def _detect_momentary_cessation(self, p, v, t, t_start, t_end):
    """
    Standard: NERC PRC-029-1 - no-trip region  
    """
    p = np.asarray(p, dtype=float)  # float array; no copy if it already is one
    v = np.asarray(v, dtype=float)
    t = np.asarray(t, dtype=float)
    if q is None:
       i_mag = np.abs(p)
    else:
      q = np.asarray(q, dtype=float)
      i_mag = np.sqrt(p**2 + q**2) / np.maximum(v, 1e-3)
    mandatory = ((v >= 0.10) & (v < self.v_low)) | ((v > 1.10) & (v <= 1.20))
    blocked = (t >= t_start) & (t <= t_end) & mandatory & (i_mag < 0.05) # make judgement based on current
    dt = np.diff(t, append=t[-1])           # time each sample stands for
    longest, run = 0.0, 0.0
    for k in range(len(t)):
      if blocked[k]:
        run += dt[k]                        # extend the current blocked run
        longest = max(longest, run)
      else:
        run = 0.0                           # run ends
    return longest

  def analyze_fault(self, v, t, p, f):
    """
    Assemble ride-through measures
    """
    v = np.asarray(v, dtype=float); t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float); f = np.asarray(f, dtype=float)
    t_start, t_end = self._detect_fault_window(v, t)
    if t_start is None:
        return {"fault_detected": {"value": False, "unit": "", "desc_zh": "No fault"}}
    lv = self._band_dwell(v, t, self.lvrt_bands)
    hv = self._band_dwell(v, t, self.hvrt_bands)
    fr = self._band_dwell(f, t, self.freq_bands)
    key_map = {"V<0.25": "lvrt_dwell_Vlt0p25_s", "0.25-0.50": "lvrt_dwell_0p25leVlt0p50_s",
               "0.50-0.70": "lvrt_dwell_0p50leVlt0p70_s", "0.70-0.90": "lvrt_dwell_0p70leVlt0p90_s",
               "1.10-1.20": "hvrt_dwell_1p10Vlt1p20_s",
               "57.0-58.8": "freq_dwell_57p0leflt58p8_s", "61.2-61.8": "freq_dwell_61p2ltflt61p8_s"}

    def _entry(val, unit, zh): # zh: desc_zh, pre-defined data format protocol 
        return {"value": val, "unit": unit, "desc_zh": zh}
    
    out = {}
    for d in lv + hv + fr:
        out[key_map[d["tag"]]] = _entry(d["dwell_s"], "s", "Dwell [%s]" % d["tag"])
    out["ride_through_worst_margin_s"] = _entry(
        min(d["margin_s"] for d in lv + hv + fr), "s", "Worst ride-through margin")
    out["lvrt_zero_voltage_duration_wind_s"] = _entry(lv[0]["dwell_s"], "s", "Zero-voltage dwell")
    out["hvrt_110_120pct_duration_s"] = _entry(hv[0]["dwell_s"], "s", "HVRT dwell")
    out["fault_voltage_dip_pu"] = _entry(float(np.min(v)), "pu", "Voltage dip")
    out["fault_duration_s"] = _entry(t_end - t_start, "s", "Fault duration")
    out["active_power_recovery_time_s"] = _entry(
        self._recovery_time(v, p, t, t_start, t_end), "s", "P recovery time")
    out["momentary_cessation_detected"] = _entry(
        self._detect_momentary_cessation(p, v, t, t_start, t_end), "", "Cessation flag")
    return out
