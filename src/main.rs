use arrayfire as af;
use arrayfire::{
    dim4, exp, Array, Dim4, MatProp, RandomEngine, RandomEngineType, DEFAULT_RANDOM_ENGINE,
};
use image::ColorType;
use mnist::{MnistBuilder, NormalizedMnist};
use std::ops::Mul;
use std::sync::Arc;

fn main() {
    arrayfire::info();

    let mut network = Network::new(28 * 28)
        .add_layer(400, Activation::RELU)
        .add_layer(200, Activation::RELU)
        .add_layer(10, Activation::SIGMOID);

    let mnist = MnistBuilder::new()
        .label_format_one_hot()
        .finalize()
        .normalize();

    for i in 0..1 {
        let mut imgs = mnist
            .trn_img
            .chunks_exact(28 * 28)
            .map(|img| {
                let img = img.iter().map(|v| *v as f64).collect::<Vec<_>>();
                Array::new(&img, dim4!(28 * 28))
            })
            .peekable();

        let mut labels = mnist.trn_lbl.chunks_exact(10).map(|lbl| {
            let lbl = lbl.iter().map(|v| *v as f64).collect::<Vec<_>>();
            Array::new(&lbl, dim4!(10))
        });

        let batch_size = 128;
        let mut i = 0;
        while imgs.peek().is_some() {
            i += 1;
            let mut batch_img = Vec::with_capacity(batch_size);
            let mut batch_lbl = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                if let Some(img) = imgs.next() {
                    batch_img.push(img);
                }
                if let Some(lbl) = labels.next() {
                    batch_lbl.push(lbl);
                }
            }
            print!("minibatch {i} - ");

            network.learn(&batch_img, &batch_lbl, 0.1);

            let (error, accuracy) = validate(&network, &mnist);
            println!("error = {:.2} - accuracy = {:.1}%", error, accuracy);
        }
    }

    does_it_work(&network, &mnist);
}

fn validate(net: &Network, mnist: &NormalizedMnist) -> (f64, f64) {
    let target_n = 400;

    let mut error = 0.0;
    let mut n = 0;
    let mut correct = 0;

    let imgs = mnist
        .tst_img
        .chunks_exact(28 * 28)
        .map(|img| {
            let img = img.iter().map(|v| *v as f64).collect::<Vec<_>>();
            Array::new(&img, dim4!(28 * 28))
        })
        .take(target_n);
    let labels = mnist
        .tst_lbl
        .chunks_exact(10)
        .map(|lbl| {
            let lbl = lbl.iter().map(|v| *v as f64).collect::<Vec<_>>();
            Array::new(&lbl, dim4!(10))
        })
        .take(target_n);

    for (img, lbl) in imgs.zip(labels) {
        n += 1;
        let result = net.forward(img.clone());
        let diff = &result - &lbl;
        let error = af::sum_all(&(&diff * &diff)).0;
        //error += af::sum_all(&(&diff * &diff)).0;

        let (_, _, result_idx) = af::imax_all(&result);
        let (_, _, correct_idx) = af::imax_all(&lbl);
        if result_idx == correct_idx {
            correct += 1;
        }
    }

    println!("{} out of {}", correct, n);

    let accuracy = correct as f64 / n as f64;
    (error / (n as f64 * 10.0), accuracy * 100.0)
}

fn does_it_work(net: &Network, mnist: &NormalizedMnist) {
    std::fs::remove_dir_all("./out").unwrap();
    std::fs::create_dir("./out").unwrap();

    let imgs = mnist.tst_img.chunks_exact(28 * 28);
    let lbls = mnist.tst_lbl.chunks_exact(10);

    for (i, (img, lbl)) in imgs.zip(lbls).take(30).enumerate() {
        let net_img = Array::new(
            &img.iter().map(|v| *v as f64).collect::<Vec<_>>(),
            dim4!(28 * 28),
        );
        let net_lbl = Array::new(
            &lbl.iter().map(|v| *v as f64).collect::<Vec<_>>(),
            dim4!(10),
        );
        let lbl = af::imax_all(&net_lbl).2;
        let result = af::imax_all(&net.forward(net_img)).2;

        let img_buf = img.iter().map(|v| (v * 255.0) as u8).collect::<Vec<_>>();
        image::save_buffer(
            format!("./out/{} identified as {} - {}.png", lbl, result, i),
            &img_buf,
            28,
            28,
            ColorType::L8,
        )
        .unwrap();
    }
}

struct Network {
    inputs: u64,
    layers: Vec<Layer>,
    rand: RandomEngine,
}

impl Network {
    fn new(inputs: u64) -> Self {
        Network {
            inputs,
            layers: vec![],
            rand: RandomEngine::new(DEFAULT_RANDOM_ENGINE, None),
        }
    }

    fn add_layer(mut self, size: u64, activation: Activation) -> Self {
        let inputs = self.layers.last().map(|l| l.size).unwrap_or(self.inputs);
        let layer = Layer::new(inputs, size, activation, &self.rand);
        self.layers.push(layer);
        self
    }

    fn forward(&self, mut input: Array<f64>) -> Array<f64> {
        for layer in &self.layers {
            input = layer.forward(input);
        }
        input
    }

    fn forward_with_activations(&self, mut input: Array<f64>) -> Activations {
        let mut activations = Vec::with_capacity(self.layers.len() + 1);
        activations.push(input.copy());

        for layer in &self.layers {
            input = layer.forward(input);
            activations.push(input.copy());
        }
        Activations(activations)
    }

    fn learn(&mut self, inputs: &[Array<f64>], expected: &[Array<f64>], learning_rate: f64) {
        assert_eq!(inputs.len(), expected.len());

        let batch_size = inputs.len();
        let mut avg_gradients = {
            let mut vec = vec![];
            for i in 0..self.layers.len() {
                let dim = self.layers[i].weights.dims();
                vec.push(af::constant(0f64, dim).copy());
            }
            vec
        };
        let mut avg_deltas = {
            let mut vec = vec![];
            for i in 0..self.layers.len() {
                let dim = self.layers[i].biases.dims();
                vec.push(af::constant(0f64, dim).copy());
            }
            vec
        };

        for (input, expected) in inputs.into_iter().zip(expected.into_iter()) {
            let activations = self.forward_with_activations(input.clone());
            let deltas = self.calculate_deltas(expected, &activations);
            let gradients = self.calculate_gradients(&activations, &deltas);

            for layer_idx in 0..self.layers.len() {
                avg_gradients[layer_idx] = &avg_gradients[layer_idx] + &gradients[layer_idx];
                avg_deltas[layer_idx] = &avg_deltas[layer_idx] + &deltas[layer_idx];
            }
        }

        for i in 0..self.layers.len() {
            // average gradients & deltas
            avg_gradients[i] = &avg_gradients[i] / batch_size as f64;
            avg_deltas[i] = &avg_deltas[i] / batch_size as f64;

            // apply
            let layer = &mut self.layers[i];
            layer.weights -= learning_rate * &avg_gradients[i];
            layer.biases -= learning_rate * &avg_deltas[i];
        }
    }

    fn calculate_gradients(
        &self,
        activations: &Activations,
        deltas: &[Array<f64>],
    ) -> Vec<Array<f64>> {
        deltas
            .iter()
            .enumerate()
            .map(|(layer, delta)| {
                let prev_a = &activations.for_layer(layer as i64 - 1);

                af::matmul(
                    delta,
                    &af::transpose(prev_a, false),
                    MatProp::NONE,
                    MatProp::NONE,
                )
            })
            .collect()
    }

    // calculate deltas for every layer
    fn calculate_deltas(
        &self,
        expected: &Array<f64>,
        activations: &Activations,
    ) -> Vec<Array<f64>> {
        let mut deltas = vec![];

        // adjust last layer
        let mut next_delta = {
            let layer_idx = self.layers.len() - 1;
            let layer = &self.layers[layer_idx];

            let prev_a = &activations.for_layer(layer_idx as i64 - 1);

            (activations.for_layer(layer_idx as i64) - expected)
                * layer.activation.d_vectorized(&af::matmul(
                    &layer.weights,
                    prev_a,
                    MatProp::NONE,
                    MatProp::NONE,
                ))
        };
        deltas.push(next_delta.copy());

        for i in (0..(self.layers.len() - 1)).rev() {
            let layer = &self.layers[i];
            let weights = &layer.weights;
            let next_layer = &self.layers[i + 1];
            let prev_a = &activations.for_layer(i as i64 - 1);
            let next_weights = &next_layer.weights;

            let delta = af::matmul(
                &af::transpose(next_weights, false),
                &next_delta,
                MatProp::NONE,
                MatProp::NONE,
            ) * layer.activation.d_vectorized(&af::matmul(
                weights,
                prev_a,
                MatProp::NONE,
                MatProp::NONE,
            ));

            deltas.push(delta.copy());
            next_delta = delta;
        }

        deltas.reverse();
        deltas
    }
}

struct Activations(Vec<Array<f64>>);

impl Activations {
    fn for_layer(&self, layer: i64) -> &Array<f64> {
        &self.0[(layer + 1) as usize]
    }
}

struct Layer {
    inputs: u64,
    size: u64,
    weights: Array<f64>,
    biases: Array<f64>,
    activation: Activation,
}

impl Layer {
    fn new(inputs: u64, size: u64, activation: Activation, rand: &RandomEngine) -> Self {
        let weights = af::random_normal(dim4!(size, inputs), rand); // TODO: is this the right dim?
        let biases = af::constant(0.0, dim4!(size));
        Layer {
            inputs,
            size,
            weights,
            biases,
            activation,
        }
    }

    fn forward(&self, inputs: Array<f64>) -> Array<f64> {
        let z = af::matmul(&self.weights, &inputs, MatProp::NONE, MatProp::NONE) + &self.biases;
        self.activation.vectorized(&z)
    }
}

#[derive(Copy, Clone)]
enum Activation {
    SIGMOID,
    RELU,
    SOFTMAX,
}

impl Activation {
    fn vectorized(self, x: &Array<f64>) -> Array<f64> {
        match self {
            Self::SIGMOID => 1.0 / (1.0 + exp(&(-1.0 * x))),
            Self::RELU => af::clamp(x, &0f64, &f64::INFINITY, true),
            Self::SOFTMAX => {
                let exp = exp(x);
                let (sum, _) = af::sum_all(&exp);
                exp / sum
            }
            _ => unimplemented!(),
        }
    }

    fn d_vectorized(self, x: &Array<f64>) -> Array<f64> {
        match self {
            Self::SIGMOID => {
                let f = self.vectorized(x);
                &f * (1.0 - &f)
            }
            Self::RELU => af::ge(x, &0f64, true).cast(),
            Self::SOFTMAX => af::constant(1.0, x.dims()),
            _ => unimplemented!(),
        }
    }
}
