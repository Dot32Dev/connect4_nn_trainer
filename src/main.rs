use nalgebra::{DMatrix, DVector};
use rand::distributions::{Distribution, Uniform};

struct NeuralNetwork {
    layers: Vec<DMatrix<f64>>, // Weights for each layer
    biases: Vec<DVector<f64>>, // Biases for each layer
}

impl NeuralNetwork {
    fn new(layer_sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        let mut biases = Vec::new();

        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(-1.0, 1.0);

        for i in 0..layer_sizes.len() - 1 {
            // // Randomly initialize weights and biases
            // let layer =
            //     DMatrix::<f64>::new_random(layer_sizes[i + 1], layer_sizes[i]);

            // Randomly initialize weights and biases
            let layer =
                DMatrix::from_fn(layer_sizes[i + 1], layer_sizes[i], |_, _| {
                    uniform.sample(&mut rng)
                });

            // let bias = DVector::<f64>::new_random(layer_sizes[i + 1]);

            let bias = DVector::from_fn(layer_sizes[i + 1], |_, _| {
                uniform.sample(&mut rng)
            });
            layers.push(layer);
            biases.push(bias);
        }

        NeuralNetwork { layers, biases }
    }

    fn forward(&self, input: &DVector<f64>) -> DVector<f64> {
        let mut output = input.clone();
        for (layer, bias) in self.layers.iter().zip(self.biases.iter()) {
            output = layer * output + bias; // Matrix multiplication and bias addition
            output = output.map(|x| x.max(0.0)); // Apply ReLU activation
        }
        output
    }
}

fn bitboard_to_input(player1: u64, player2: u64) -> DVector<f64> {
    let mut input = DVector::zeros(42); // 42 playable positions (6x7 grid)

    for i in 0..42 {
        let mask = 1u64 << i;
        if player1 & mask != 0 {
            input[i] = -1.0;
        } else if player2 & mask != 0 {
            input[i] = 1.0;
        }
        // If neither player has a piece here, it remains 0.0
    }

    input
}

fn main() {
    let network = NeuralNetwork::new(&[42, 64, 64, 7]);
}
