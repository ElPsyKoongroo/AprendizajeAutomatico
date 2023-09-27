use serde::de::DeserializeOwned;
use std::fs::read_to_string;

static ITERS: usize = 1000;
static ALPHA: f64 = 0.0031;

fn read_csv<T>(path: impl AsRef<std::path::Path>) -> Result<Vec<T>, csv::Error>
where
    T: Sized + DeserializeOwned,
{
    let raw_data = read_to_string(path)?;

    let mut parsed_data: Vec<T> = vec![];
    let mut csv = csv::Reader::from_reader(raw_data.as_bytes());

    for entry in csv.deserialize() {
        let entry: T = entry?;
        parsed_data.push(entry);
    }

    Ok(parsed_data)
}

fn hipotesis(tita: f64, row: &[f64]) -> f64 {
    row.iter()
        .map(|x| x * tita)
        .reduce(|acc, val| acc + val)
        .unwrap_or_default()
}

fn main() {
    let data_set = read_csv::<Vec<f64>>("./regresion_1.csv").expect("No se pudo leer el fichero");

    let mut titas = [1.0, 2.13];
    for _ in 0..ITERS {
        for i in 0..titas.len() {
            for entry in data_set.iter() {
                let h = hipotesis(titas[i], &entry[0..1]);
                titas[i] = titas[i] - ALPHA * (h - entry[1]) * entry[i]
            }
        }
    }
}
