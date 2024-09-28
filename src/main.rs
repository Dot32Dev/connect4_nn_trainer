use connect4_board_library::Bitboard;
use nalgebra::{DMatrix, DVector};
use rand::distributions::{Distribution, Uniform};
use rand::Rng;

trait Participant {
    fn make_move(&self, game: &mut Bitboard, you: usize, opponent: usize);
}

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

fn sort_indices_by_values(arr: &DVector<f64>) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..arr.len()).collect();

    // Sort the indices based on the corresponding values in the DVector, in
    // reverse for largest first
    indices.sort_by(|&i, &j| arr[j].partial_cmp(&arr[i]).unwrap());

    indices
}

impl Participant for NeuralNetwork {
    fn make_move(&self, game: &mut Bitboard, you: usize, opponent: usize) {
        let input =
            bitboard_to_input(game.players[you], game.players[opponent]);
        let output = sort_indices_by_values(&self.forward(&input));
        for i in 0..7 {
            if game.drop_piece(output[i]) {
                return;
            }
        }
    }
}

struct RandomAgent;

impl Participant for RandomAgent {
    fn make_move(&self, game: &mut Bitboard, _you: usize, _opponent: usize) {
        while !game.drop_piece(rand::thread_rng().gen_range(0..7)) {}
    }
}

struct Tournament {
    participants: Vec<Vec<Box<dyn Participant>>>,
}

impl Tournament {
    fn new_random(num_participants: usize) -> Self {
        let mut participants = vec![vec![]];

        // Add Neural Networks
        for _ in 0..num_participants - 1 {
            participants[0]
                .push(Box::new(NeuralNetwork::new(&[42, 64, 64, 7]))
                    as Box<dyn Participant>);
        }

        // Add one RandomAgent
        participants[0].push(Box::new(RandomAgent) as Box<dyn Participant>);

        Self { participants }
    }

    fn run_tournament(&mut self) {
        let mut round = 0;

        while self.participants[round].len() > 1 {
            // If the amount of participants is uneven, randomly move a participant
            // to the next round.
            let length = self.participants[round].len();
            if length % 2 != 0 {
                // Remove a random participant
                let bye_index = rand::thread_rng().gen_range(0..length);
                let bye = self.participants[round].remove(bye_index);

                // If there is no next round, create it
                if self.participants.len() <= round + 1 {
                    self.participants.push(Vec::new());
                }

                // Add the removed participant to the next round
                self.participants[round + 1].push(bye);
            }

            // Iterate over two AI's at a time and verse them together.
            let mut full_idx = 0; // So that we can make the right person win.
            let mut winners = Vec::new(); // Avoids multiple mutable borrow issues.

            for pair in self.participants[round].chunks_exact(2) {
                let mut game = Bitboard::new();

                'game: loop {
                    let you = (game.move_counter & 1);
                    let opponent = 1 - (game.move_counter & 1);

                    pair[you].make_move(&mut game, you, opponent);

                    if game.check_win() {
                        winners.push(full_idx + you);

                        // Print the last round
                        if self.participants[round].len() == 2 {
                            println!("{}", game);
                        }

                        break 'game;
                    }

                    if game.move_counter >= 42 {
                        // Oh well, promote both to the next round:
                        winners.push(full_idx + you);
                        winners.push(full_idx + opponent);
                        break 'game;
                    }

                    // Print the last round
                    if self.participants[round].len() == 2 {
                        println!("{}", game);
                    }
                }

                full_idx += 2;
            }

            // Move winners to the next round
            if self.participants.len() <= round + 1 {
                self.participants.push(Vec::new());
            }
            for winner_idx in winners.into_iter().rev() {
                let winner = self.participants[round].remove(winner_idx);
                self.participants[round + 1].push(winner);
            }

            // Advance to the next round
            round += 1;
        }
    }
}

fn main() {
    let mut tournament = Tournament::new_random(100);
    tournament.run_tournament();

    // let network = NeuralNetwork::new(&[42, 64, 64, 7]);
    // let mut game = Bitboard::new();

    // loop {
    //     let ai_1 = (game.move_counter & 1);
    //     let ai_2 = 1 - (game.move_counter & 1);

    //     network.select_move(&mut game, ai_1, ai_2);

    //     if game.check_win() {
    //         break;
    //     }

    //     if game.move_counter > 42 {
    //         break;
    //     }
    // }

    // println!("{}", game);
}
