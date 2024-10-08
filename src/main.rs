use connect4_board_library::Bitboard;
use nalgebra::{DMatrix, DVector};
use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::any::type_name;
use std::any::TypeId;
use std::fmt;
use std::fs::OpenOptions;
use std::fs::{self, File};
use std::io::BufWriter;
use std::io::{Read, Write};
use std::path::Path;
use std::time::{Duration, Instant};

#[typetag::serde(tag = "type")]
trait Participant: std::fmt::Debug {
    fn make_move(&self, game: &mut Bitboard, you: usize, opponent: usize);
    fn type_id(&self) -> TypeId;
}

#[derive(Serialize, Deserialize)]
struct NeuralNetwork {
    id: u8,
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

        NeuralNetwork {
            id: rng.gen::<u8>(),
            layers,
            biases,
        }
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

#[typetag::serde]
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

    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }
}

fn type_name_of<T>(_: T) -> &'static str {
    type_name::<T>()
}

impl fmt::Debug for NeuralNetwork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // All I really want it to print is the type name
        write!(
            f,
            "{}, id: {}",
            type_name_of(NeuralNetwork {
                id: 0,
                layers: Vec::new(),
                biases: Vec::new()
            }),
            self.id
        )
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct RandomAgent;

#[typetag::serde]
impl Participant for RandomAgent {
    fn make_move(&self, game: &mut Bitboard, _you: usize, _opponent: usize) {
        while !game.drop_piece(rand::thread_rng().gen_range(0..7)) {}
    }

    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct TowerAgent {
    columns: [usize; 7],
}

impl TowerAgent {
    fn new() -> Self {
        let mut columns = [0, 1, 2, 3, 4, 5, 6];
        let mut rng = rand::thread_rng();
        columns.shuffle(&mut rng);
        TowerAgent { columns }
    }
}

#[typetag::serde]
impl Participant for TowerAgent {
    fn make_move(&self, game: &mut Bitboard, _you: usize, _opponent: usize) {
        for i in 0..7 {
            if game.drop_piece(self.columns[i]) {
                return;
            }
        }
    }

    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }
}

// fn is_special_agent(participant: &Box<dyn Participant>) -> bool {
//     participant.type_id() == TypeId::of::<TowerAgent>()
//         || participant.type_id() == TypeId::of::<RandomAgent>()
// }

#[derive(Serialize, Deserialize, Debug)]
struct Tournament {
    // Generation keeps track of how many tournaments have happened in the past.
    generation: u32,
    // The real world time the models have spent in training.
    training_time: Duration,
    participants: Vec<Vec<Box<dyn Participant>>>,
}

impl Tournament {
    fn new_random(num_participants: usize) -> Self {
        let mut participants = vec![vec![]];

        // Add Neural Networks (Subtracting 2 to account for other AIs)
        for _ in 0..num_participants - 2 {
            participants[0]
                .push(Box::new(NeuralNetwork::new(&[42, 64, 64, 7]))
                    as Box<dyn Participant>);
        }

        // Add one RandomAgent
        participants[0].push(Box::new(RandomAgent) as Box<dyn Participant>);

        // Add one TowerAgent
        participants[0]
            .push(Box::new(TowerAgent::new()) as Box<dyn Participant>);

        Self {
            generation: 0,
            training_time: Duration::new(0, 0),
            participants,
        }
    }

    fn run_tournament(&mut self) {
        let start = Instant::now();

        let mut round = 0;

        while self.participants[round].len() > 1 {
            let mut rng = rand::thread_rng();
            self.participants[round].shuffle(&mut rng);
            // If the amount of participants is uneven, randomly move a participant
            // to the next round.
            let length = self.participants[round].len();
            if length % 2 != 0 {
                // Remove a random participant
                let bye_index = rng.gen_range(0..length);
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
                let mut game_count = 0;
                let mut player_wins = [0; 2];
                for _ in 0..2 {
                    let mut game = Bitboard::new();

                    'game: loop {
                        let mut you = game.move_counter & 1;
                        let mut opponent = 1 - (game.move_counter & 1);

                        if game_count == 1 {
                            you = 1 - (game.move_counter & 1);
                            opponent = game.move_counter & 1;
                        }

                        pair[you].make_move(&mut game, you, opponent);

                        if game.check_win() {
                            // winners.push(full_idx + you);
                            player_wins[you] += 1;

                            // Print the last round
                            if self.participants[round].len() == 2 {
                                println!(
                                    "Game {} winner: {:?}\n loser {:?}",
                                    game_count + 1,
                                    self.participants[round][full_idx + you],
                                    self.participants[round]
                                        [full_idx + opponent]
                                );
                                println!("{}", game);
                            }

                            break 'game;
                        }

                        if game.move_counter >= 42 {
                            // Oh well, promote both to the next round:
                            // winners.push(full_idx + you);
                            // winners.push(full_idx + opponent);
                            player_wins[you] += 1;
                            player_wins[opponent] += 1;
                            break 'game;
                        }

                        // // Print the last round
                        // if self.participants[round].len() == 2 {
                        //     println!("{}", game);
                        // }
                    }
                    game_count += 1;
                }
                if player_wins[0] > player_wins[1] {
                    winners.push(full_idx + 0);
                } else if player_wins[0] == player_wins[1] {
                    // Actually, maybe drawed players should both not go higher
                    // winners.push(full_idx + 0);
                    // winners.push(full_idx + 1);
                } else {
                    winners.push(full_idx + 1);
                }

                full_idx += 2;
            }

            // Move winners to the next round
            if self.participants.len() <= round + 1 {
                self.participants.push(Vec::new());
            }
            winners.sort_unstable();
            for winner_idx in winners.into_iter().rev() {
                let winner = self.participants[round].remove(winner_idx);
                self.participants[round + 1].push(winner);
            }

            // Advance to the next round
            round += 1;
        }

        let duration = start.elapsed();
        self.generation += 1;
        self.training_time += duration;
    }

    fn stats(&self) {
        println!("Generation {}", self.generation);
        println!("The number of rounds was: {}", self.participants.len());
        println!("{} seconds spent in training", self.training_time.as_secs());
        for i in 0..self.participants.len() {
            println!(
                "{} eliminated in round {}",
                self.participants[i].len(),
                i
            );
        }
    }

    fn find_special_agents(&self) -> (usize, usize) {
        let mut tower_agent_round = 0;
        let mut random_agent_round = 0;

        for (i, round) in self.participants.iter().enumerate() {
            for agent in round {
                if agent.type_id() == TypeId::of::<TowerAgent>() {
                    tower_agent_round = i;
                }
                if agent.type_id() == TypeId::of::<RandomAgent>() {
                    random_agent_round = i;
                }
            }
        }

        return (
            self.find_participants_below_round(tower_agent_round),
            self.find_participants_below_round(random_agent_round),
        );
    }

    fn find_participants_below_round(&self, target_round: usize) -> usize {
        let mut participants = 0;
        for round in self.participants.iter().take(target_round) {
            for _ in round {
                participants += 1;
            }
        }
        return participants;
    }

    fn flatten_participants(&mut self) {
        let mut flattened = Vec::new();

        // Move all participants from all rounds into the flattened vector
        for round in self.participants.drain(..) {
            flattened.extend(round);
        }

        // Replace the old participants structure with the flattened version
        self.participants = vec![flattened];
    }
}

fn save_tournament(tournament: &Tournament, folder: &Path) {
    let filename = format!("Generation {}.ron", tournament.generation);
    let filepath = folder.join(filename);

    let serialized = ron::ser::to_string_pretty(
        tournament,
        ron::ser::PrettyConfig::default(),
    )
    .expect("Failed to serialize tournament");

    let mut file = File::create(filepath).expect("Failed to create file");
    file.write_all(serialized.as_bytes())
        .expect("Failed to write to file");
}

fn load_tournament(filepath: impl AsRef<Path>) -> Tournament {
    let mut file = File::open(filepath).expect("Failed to open file");
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .expect("Failed to read file");

    ron::from_str(&contents).expect("Failed to deserialize tournament")
}

fn save_special_agents(agents: Vec<(usize, usize)>, generation: u32) {
    // Open the file for appending, create if it doesn't exist
    let file = OpenOptions::new()
        .append(true)
        .create(true)
        .open("special_agents.csv")
        .unwrap();
    let mut writer = BufWriter::new(file);

    // If the CSV was empty, add the column names
    if writer.get_ref().metadata().unwrap().len() == 0 {
        writeln!(writer, "Generation, Tower Agent, Random Agent").unwrap();
    }

    // Append agents with generation number
    let length = agents.len();
    for (i, (tower_agent, random_agent)) in agents.into_iter().enumerate() {
        let gen = generation as usize - (length - 1 - i); // Decrease generation
        writeln!(writer, "{}, {}, {}", gen, tower_agent, random_agent).unwrap();
    }
}

fn main() {
    let history_folder = Path::new("network_history");
    let mut tournament: Tournament;

    // Create the network_history folder if it doesn't exist
    fs::create_dir_all(history_folder)
        .expect("Failed to create network_history folder");

    // Check if the folder is empty
    let mut entries = fs::read_dir(history_folder)
        .expect("Failed to read network_history folder")
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to collect directory entries");
    entries.sort_by_key(|entry| entry.file_name());

    if entries.is_empty() {
        // Create a new tournament if the folder is empty
        tournament = Tournament::new_random(100);
        save_tournament(&tournament, history_folder);
    } else {
        // Load the most recent generation
        let latest_file = entries.last().unwrap();
        tournament = load_tournament(latest_file.path());
    }

    loop {
        // Run for one minute before saving to file
        let start_time = Instant::now();
        let duration = Duration::from_secs(60);

        // Keep track of how well special agents do each round, only save them
        // to file after the one minute has finished.
        let mut special_agents: Vec<(usize, usize)> = Vec::new();

        // Run generations
        while start_time.elapsed() < duration {
            tournament.flatten_participants();
            tournament.run_tournament();
            tournament.stats();
            special_agents.push(tournament.find_special_agents());
        }

        // Save data to file
        save_tournament(&tournament, history_folder);
        save_special_agents(special_agents, tournament.generation);
    }
}
