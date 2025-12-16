use cgmath::{InnerSpace, Matrix4, Point3, Vector3};

pub struct Camera {
    pub position: Point3<f32>,
    pub target: Point3<f32>,
    pub up: Vector3<f32>,
    pub yaw: f32,
    pub pitch: f32,
    pub fovy: f32,
    pub aspect: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    pub fn new(width: u32, height: u32) -> Self {
        let position: Point3<f32> = Point3::new(0.0, 5.0, 10.0);
        let target: Point3<f32> = Point3::new(0.0, 0.0, 0.0);
        let forward = (target - position).normalize();
        let yaw: f32 = forward.z.atan2(forward.x);
        let pitch: f32 = forward.y.asin();
        Self {
            position,
            target,
            up: Vector3::new(0.0, 1.0, 0.0),
            yaw,
            pitch,
            fovy: 45.0,
            aspect: width as f32 / height as f32,
            znear: 0.1,
            zfar: 1000.0,
        }
    }

    pub fn set_look_at(&mut self, position: Point3<f32>, target: Point3<f32>) {
        self.position = position;
        self.target = target;
        let forward = (self.target - self.position).normalize();
        self.yaw = forward.z.atan2(forward.x);
        self.pitch = forward.y.asin();
    }

    pub fn forward(&self) -> Vector3<f32> {
        Vector3::new(
            self.pitch.cos() * self.yaw.cos(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.sin(),
        )
        .normalize()
    }

    pub fn right(&self) -> Vector3<f32> {
        self.forward().cross(self.up).normalize()
    }

    pub fn apply_mouse_look(&mut self, dx: f32, dy: f32, sensitivity: f32) {
        self.yaw += dx * sensitivity;
        self.pitch -= dy * sensitivity;
        let limit = 1.553343;
        if self.pitch > limit {
            self.pitch = limit;
        }
        if self.pitch < -limit {
            self.pitch = -limit;
        }
        let f = self.forward();
        self.target = self.position + f;
    }

    pub fn move_fly(&mut self, dir: Vector3<f32>, dt: f32, speed: f32) {
        let delta = dir * (speed * dt);
        self.position += delta;
        self.target += delta;
    }
    
    pub fn update_aspect(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }
    
    pub fn view_matrix(&self) -> Matrix4<f32> {
        Matrix4::look_at_rh(self.position, self.target, self.up)
    }
    
    pub fn projection_matrix(&self) -> Matrix4<f32> {
        cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub view_inv: [[f32; 4]; 4],
    pub proj_inv: [[f32; 4]; 4],
    pub position: [f32; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: Matrix4::from_scale(1.0).into(),
            view_inv: Matrix4::from_scale(1.0).into(),
            proj_inv: Matrix4::from_scale(1.0).into(),
            position: [0.0, 0.0, 0.0, 1.0],
        }
    }
    
    pub fn update(&mut self, camera: &Camera) {
        use cgmath::SquareMatrix;
        
        let view = camera.view_matrix();
        let proj = camera.projection_matrix();
        let view_proj = proj * view;
        
        self.view_proj = view_proj.into();
        self.view_inv = view.invert().unwrap().into();
        self.proj_inv = proj.invert().unwrap().into();
        self.position = [camera.position.x, camera.position.y, camera.position.z, 1.0];
    }
}
