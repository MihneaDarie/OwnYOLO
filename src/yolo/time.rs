use std::sync::OnceLock;
use std::time::Instant;

#[allow(dead_code)]
static GLOBAL_TIMER: OnceLock<Instant> = OnceLock::new();

#[allow(dead_code)]
fn get_global_timer() -> &'static Instant {
    GLOBAL_TIMER.get_or_init(Instant::now)
}

#[allow(dead_code)]
pub fn print_time(message: &str) {
    let seconds = get_global_timer().elapsed().as_micros();
    println!("{message}: {seconds}");
}
