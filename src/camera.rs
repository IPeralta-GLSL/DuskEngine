use cgmath::{Matrix4, Point3, Vector3};

pub struct Camera {
    pub position: Point3<f32>,
    pub target: Point3<f32>,
    pub up: Vector3<f32>,
    pub fovy: f32,
    pub aspect: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            position: Point3::new(0.0, 5.0, 10.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            fovy: 45.0,
            aspect: width as f32 / height as f32,
            znear: 0.1,
            zfar: 1000.0,
        }
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
