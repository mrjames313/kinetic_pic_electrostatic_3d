use anyhow::Result;

pub struct SingleDimSpec {
    n: usize,
    min: f64,
    max: f64,
    // computed fields
    delta: f64,
    center: f64,
}

impl SingleDimSpec {
    pub fn init(n: usize, min: f64, max: f64) -> Result<Self> {
        let spec = Self{n:n, min:min, max:max,
                            delta:(max - min)/(n-1) as f64,
                            center:(max - min) / 2.0 };
        Ok(spec)
    }

    pub fn print(&self) -> Result <()> {
        println!("Extent: [{:.4}, {:.4}], {} cells", self.min, self.max, self.n-1);
        Ok(())
    }
}

pub struct ThreeDWorldSpec {
    x_dim: SingleDimSpec,
    y_dim: SingleDimSpec,
    z_dim: SingleDimSpec,
}

impl ThreeDWorldSpec {
    
    pub fn init(x_dim: SingleDimSpec, y_dim: SingleDimSpec, z_dim: SingleDimSpec) -> Result<Self> {
        let spec = Self{x_dim:x_dim, y_dim:y_dim, z_dim:z_dim};
        Ok(spec)
    }

    pub fn print(&self) -> Result <()> {
        println!("Three dimensional world mesh with dimensions:");
        print!("X: ");
        self.x_dim.print()?;
        print!("Y: ");
        self.y_dim.print()?;
        print!("Z: ");
        self.z_dim.print()?;
        Ok(())
    }
}
    
