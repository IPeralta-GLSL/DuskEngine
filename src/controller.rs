use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

pub struct InputState {
    pub forward: bool,
    pub back: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
    pub sprint: bool,
    pub mouse_captured: bool,
    pub mouse_delta: (f32, f32),
}

impl InputState {
    pub fn new() -> Self {
        Self {
            forward: false,
            back: false,
            left: false,
            right: false,
            up: false,
            down: false,
            sprint: false,
            mouse_captured: false,
            mouse_delta: (0.0, 0.0),
        }
    }

    pub fn on_window_event(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::KeyW) => self.forward = pressed,
                    PhysicalKey::Code(KeyCode::KeyS) => self.back = pressed,
                    PhysicalKey::Code(KeyCode::KeyA) => self.left = pressed,
                    PhysicalKey::Code(KeyCode::KeyD) => self.right = pressed,
                    PhysicalKey::Code(KeyCode::Space) => self.up = pressed,
                    PhysicalKey::Code(KeyCode::ControlLeft) | PhysicalKey::Code(KeyCode::ControlRight) => self.down = pressed,
                    PhysicalKey::Code(KeyCode::ShiftLeft) | PhysicalKey::Code(KeyCode::ShiftRight) => self.sprint = pressed,
                    _ => {}
                }
                true
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if *button == MouseButton::Left && *state == ElementState::Pressed {
                    self.mouse_captured = true;
                    return true;
                }
                false
            }
            _ => false,
        }
    }

    pub fn on_mouse_motion(&mut self, delta: (f64, f64)) {
        if self.mouse_captured {
            self.mouse_delta.0 += delta.0 as f32;
            self.mouse_delta.1 += delta.1 as f32;
        }
    }

    pub fn take_mouse_delta(&mut self) -> (f32, f32) {
        let d = self.mouse_delta;
        self.mouse_delta = (0.0, 0.0);
        d
    }
}
