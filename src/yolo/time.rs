use std::sync::OnceLock;
use std::time::Instant;

static GLOBAL_TIMER: OnceLock<Instant> = OnceLock::new();

fn get_global_timer() -> &'static Instant {
    GLOBAL_TIMER.get_or_init(Instant::now)
}

pub fn print_time(message: &str) {
    let seconds = get_global_timer().elapsed().as_secs_f64();
    println!("{message}: {seconds}");
}