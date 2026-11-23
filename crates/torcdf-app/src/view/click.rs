//! A click on the map or image opens a pop-up only once it is clear that no
//! second click follows (a double-click just fits the view).

use std::time::{Duration, Instant};

/// What a pending click should do at time `now`.
#[derive(Debug, PartialEq, Eq)]
pub enum ClickStep {
    /// Nothing pending.
    Idle,
    /// Still within the double-click time; look again after this long.
    Wait(Duration),
    /// No second click came: open the pop-up.
    Open,
}

/// Decision for a click made at `last_click`, as of `now`.
pub fn click_decision(now: Instant, last_click: Option<Instant>, double_click_time: f64) -> ClickStep {
    let Some(at) = last_click else { return ClickStep::Idle };
    let window = Duration::from_secs_f64(double_click_time.max(0.0));
    let age = now.saturating_duration_since(at);
    if age >= window { ClickStep::Open } else { ClickStep::Wait(window - age) }
}

/// The click waiting to open a pop-up, with its target.
#[derive(Debug, Clone, Default)]
pub struct PendingClick<T> {
    pending: Option<(Instant, T)>,
}

impl<T> PendingClick<T> {
    /// A single click happened at `now` on `target` (replaces an earlier one).
    pub fn click(&mut self, now: Instant, target: T) {
        self.pending = Some((now, target));
    }

    /// A double-click: the pending click was the first of a pair.
    pub fn cancel(&mut self) {
        self.pending = None;
    }

    /// The target to open, once the double-click time has passed; else how
    /// long to wait.
    pub fn poll(&mut self, now: Instant, double_click_time: f64) -> Result<T, ClickStep> {
        let step = click_decision(now, self.pending.as_ref().map(|p| p.0), double_click_time);
        match step {
            ClickStep::Open => Ok(self.pending.take().map(|p| p.1).expect("pending click")),
            other => Err(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn double_click_does_not_open_popup() {
        let t0 = Instant::now();
        let mut p = PendingClick::default();
        // A lone click opens only after the double-click time.
        p.click(t0, (1, 2));
        assert!(matches!(p.poll(t0 + Duration::from_millis(100), 0.3), Err(ClickStep::Wait(_))));
        assert_eq!(p.poll(t0 + Duration::from_millis(300), 0.3), Ok((1, 2)));
        assert_eq!(p.poll(t0 + Duration::from_secs(5), 0.3), Err(ClickStep::Idle));
        // A pair: the second click cancels the first; nothing ever opens.
        p.click(t0, (1, 2));
        p.cancel();
        assert_eq!(p.poll(t0 + Duration::from_secs(5), 0.3), Err(ClickStep::Idle));
    }
}
