# Deep-learning models of the ascending proprioceptive pathway are subject to illusions

<p align="center">
<img src="./assets/Fig0-landing-page.png" alt="Proprioceptive Illusions Model" width="100%"/>
</p>

## Authors
Adriana Perez Rotondo\*, Merkourios Simos\*, Florian David, Sebastian Pigeon, Olaf Blanke, & Alexander Mathis

\*These authors contributed equally to this work.

EPFL - École Polytechnique Fédérale de Lausanne, Switzerland 🇨🇭

## Overview
Proprioception is essential for perception and action. Like any other sense, proprioception is also subject to illusions. In this research, we model classic proprioceptive illusions in which tendon vibrations lead to biases in estimating the state of the body.

We investigate these illusions with task-driven models that have been trained to infer the state of the body from distributed sensory muscle spindle inputs (primary and secondary afferents). Recent work has shown that such models exhibit representations similar to the neural code along the ascending proprioceptive pathway.

Importantly, we did not train the models on illusion experiments and simulated muscle-tendon vibrations by considering their effect on primary afferents. Our results demonstrate that task-driven models are indeed susceptible to proprioceptive illusions, with the magnitude of the illusion depending on the vibration frequency. This work illustrates that primary afferents alone are sufficient to account for these classic illusions and provides a foundation for future theory-driven experiments.

Get in touch if you want to collaborate! 

The preprint is available on [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.03.15.643457v1).

The datasets are avaialable on [Zenodo](https://zenodo.org/records/14544688).

## Contributions

1. **Simulating Proprioceptive Illusions**: We modeled how tendon vibrations affect afferent signals and demonstrated that these altered inputs lead to illusory perceptions in deep learning models.

2. **Frequency-Dependent Illusion Profiles**: We showed that the strength of the illusion varies with vibration frequency, matching patterns observed in human psychophysics experiments.

3. **Role of Afferent Types**: Our research highlights distinct contributions of primary (type Ia) and secondary (type II) afferents to proprioceptive illusions.

4. **Muscle-Specific Effects**: We systematically investigated the effects of vibrating each of the 25 modeled muscles, revealing functional relationships between specific muscles and elbow angle perception.

5. **Mechanism Insights**: We established a direct link between peripheral physiological properties of muscle spindles and central perceptual illusions.

## Installation

Follow these steps to set up the environment and install the necessary packages.

### Installing the environment
1. **Create a conda environment**:
   ```bash
   conda create -n proprioception python=3.8.20
   conda activate proprioception
   ```

2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

### Downloading the data

To train or evaluate models with spindle inputs and test on FLAG3D, PCR, Ef3D and Es3D datasets the following files are required:

- `flag_pcr_training.hdf5`: train data 55k samples with 50k from PCR
- `flag_pcr_test.hdf5`: test data from FLAG3D and PCR datasets.
- `EF3D.hdf5`: Ef3D dataset with 90 elbow flexion trajectories. 
- `ES3D.hdf5`: Es3D dataset with 100 arm configurations to test vibrations.

All the data is available for download on [Zenodo](https://zenodo.org/records/14544688) under `data.zip`.

If you want to re-generate the training and test data, download the raw datasets from [huggingface](https://huggingface.co/datasets/amathislab/proprioceptive_illusions) and follow the instructions listed in `data_generation/README.md`.

### Generating spindle coefficients

The spindle coefficients used for all the models on the paper are stored in this repository: `data/spindle_coefficients` and `data/extended_spindle_coefficients`. To re-generate spindle coefficients, run the following script:

```bash
python -m extract_data.generate_spindle_coefficients
```

## Reproducing the table and figures from the paper

- We provide notebooks to generate the table and the figures from the paper. This can be done by downloading the data for the figures provided in `data_for_figs.zip` on Zenodo and running the notebooks. Or by running the vibration scripts on the provided trained models and then executing the notebooks.
- First, change the `SAVE_DIR` parameter in `directory_paths.py` to match the path where the data from Zenodo was downloaded. It should contain the directory `data_for_figs/` or the directory `trained_models/`.
- The plots are by default saved in a folder inside the model path.

### Figure 2: Statistics and model predictions

- Use the following notebooks:
  - `notebooks/figure_2/figure_2_movement_statistics.ipynb` for Panels A and B.
  - `notebooks/figure_2/figure_2_3d_trajectory_plot.ipynb`, `notebooks/figure_2/figure_2_3d_predictions_plot.ipynb`, and `notebooks/figure_2/figure_2_spinle_plot.ipynb` for Panel C.

### Figure 3: Effect of vibrations: one model example

- Use `notebooks/figure_3/fig3_vibrations_modelExample.ipynb`
- Data for figure saved in `df_fig3_vib_vary_multipleFs_addNoVibOutput_7Muscles_numCols-12.h5`
- Results from model `experiment_causal_flag-pcr_optimizedLinearFR_30k_noLayerNorm_elbowAngles_stride1_240Hz_letter_reconstruction_joints`

### Figure 4: robustness of vibration results

- Use notebook `notebooks/figure_4/fig4_vibrations_robustness.ipynb`
- Data saved in `df_all_vib_vary_vib_vary_multipleFs_sumNonVibInput_2Muscles_numCols-9.h5` for plotting.
- Results from 20 models in `experiment_causal_flag-pcr_optimized_linear_extended_5_5_letter_reconstruction_joints/`.

### Figure 5: frequency-dependent illusion

#### Panel A

- Use first part of `notebooks/figure_5/fig5_vibration_ind_aff.ipynb`
- Panel A provides an example based on the model with `coef_seed = 1`.
- Data for plots in `df_seed-1_vib_vary_ia_subChannels_1Muscles_numCols-11.pkl`

#### Panel B

- Use second part of `notebooks/figure_5/fig5_vibration_ind_aff.ipynb`
- Data for plot `df_subChannels_fig-data-withMax_rate_allSeed_vib_vary_ia_subChannels_5Muscles_numCols-5.h5`
- Data from 5 muscles, 5 coefficient seeds, 4 train seeds and 5 afferents. There are a total of 500 points in the plot.

#### Panel C

- Use `notebooks/figure_5/fig5_frequency_dependent.ipynb`
- Unconment lines to make plots for each frequency range (50, 50-100, 100, 100-150, 150-200)
- Data saved in file `df_all_vib_vary_{test_exp_dir}_2Muscles_numCols-9.h5` for each experiment name.
- Load one at a time to make the plots for tricpes and biceps vibration.
<!-- - The following folders were used for the plot:

```python
test_exp_dir = "vib_vary_multipleFs_fixFmax"
test_exp_dir = "vib_vary_multipleFs_fixFmax-100-150"
test_exp_dir = "vib_vary_multipleFs_fixFmax-50-100"
test_exp_dir = "vib_vary_multipleFs_fixFmax-150-200"
test_exp_dir = "vib_vary_multipleFs_fixFmax-50-51"
``` -->

### Figure 6: type II afferents

- Use the notebook `notebooks/figure_6/fig6_vibrations_typeII.ipynb`
- Data for fig 5 C saved as `df_all_vib_vary_vib_vary_multipleFs_vib_ia_only_2Muscles_numCols-9.h5`, `df_all_vib_vary_vib_vary_multipleFs_vib_ii_2Muscles_numCols-9.h5`, `df_all_vib_vary_vib_vary_multipleFs_vib_ii_only_2Muscles_numCols-9.h5`
- Data for fig 5 B saved as `df_train-seed-0_vib_vary_multipleFs_vib_ii_2Muscles_numCols-10.h5`

### Figure 7: vibration of other muscles

- For panel A use: `notebooks/figure_7/fig7_vibrations.ipynb`
- Data for panel A `df_all_vib_vary_vib_vary_multipleFs_otherMuscles_3Muscles_numCols-9.h5`
- For panel B use: `notebooks/figure 7/spindle_elbow_correlation.ipynb`

### Table 3: Model performance

- Use the following notebook: `notebooks/table_3/model_perf_table.ipynb`
- Results are saved in `model_accuracy_results.csv`
- Saves the table values in a csv file and prints the latex code with the results.

## Usage

### Train and test models with spindle afferent inputs

We provide 21 pre-trained models, the ones used in the paper. To train models with the same spindle coefficients of new ones follow these steps. All scripts should be executed from the main project directory. The values in the script are set to reproduce the trained models for the paper, these can be modified to train models with different spindle parameter and training seeds.

1. Download `data/cleaned_smooth` from [Zenodo](https://zenodo.org/records/14544688). Move the following files to `{BASE_DIR}/data/`:

  ```bash
      flag_pcr_training.hdf5
      flag_pcr_test.hdf5
      EF3D.hdf5
      ES3D.hdf5
  ```

2. Extract train and test datasets `extract_data/`:
    - Run `bash extract_data/generate_train_test_data.sh --data_dir {data_dir}` to generate all datasets for all models with spindle inputs in the paper. Where `data_dir` is the `{BASE_DIR}/data/`, the path to the directory with the hdf5 files downloaded from Zenodo.
    - Takes around 5 min for a single seed.
    - The new files are saved in `{data_dir}`.

3. Train model `train/`:
    - Run `bash train/train_all.sh --data_dir {data_dir}`. This script will train models with different spindle parameters and training seeds. The models are saved in `{BASE_DIR}/trained_models`.
    - Training of all 21 models takes a few hours on a GPU. We provide some pre-trained models in Zenodo that can be used directly.

4. Test model `inference/`:
    - Run `bash inference/test_models_accuracy.sh --data_dir {data_dir}`.
    - This script computes the test performance on the three test datasets: FLAG3D/PCR, Ef3D and Es3D. It saves the results as a txt file. In `{model_path}/test/{dataset_name}/accuracy.txt`

5. Test model on vibrations `test_vibrations/`:
    - To run the results for different figures of the paper run the following shell scripts:
      - `bash test_vibrations/vibrations_Fig3.sh --data_dir {data_dir}`
      - `bash test_vibrations/vibrations_Fig4.sh --data_dir {data_dir}`
      - `bash test_vibrations/vibrations_Fig5a.sh --data_dir {data_dir}`
      - `bash test_vibrations/vibrations_Fig5c.sh --data_dir {data_dir}`
      - `bash test_vibrations/vibrations_Fig6.sh --data_dir {data_dir}`
      - `bash test_vibrations/vibrations_Fig7.sh --data_dir {data_dir}`
    - These script can be used on the pre-trained models provided directly.

## Pre-trained models provided
  
- `experiment_causal_flag-pcr_optimized_linear_5_5_letter_reconstruction_joints` main example model with spindle FR inputs. Used for figure 3. This model was trained without setting directly the training seed. The coeficient seed is 0.
  - The spindle model coefficients used for each one of the coefficients seeds are obtained by using `sampled_coefficients_{i_a,ii}_5_0.csv` and `coefficients.csv` in `data/spindle_coefficients/{i_a,ii}/linear/`. The datasets to train, test and evaluate effect of vibrations on these models are the following:
  - `optimized_linear_0_5_5_{dataset_name}.hdf5` where `coef_seed` is the same number as in the model path and `dataset_name` are as defined in the above section

- `experiment_causal_flag-pcr_optimized_linear_extended_5_5_letter_reconstruction_joints` models for robustness results. All models have 5 aff type Ia and 5 aff type II. We have 5 different coefficient seeds (0,1,2,3,4) and 4 different train seeds (0,1,2,9).
  - The models are in the subdirectories of the form `spatiotemporal_4_8-8-32-64_7171_{coef_seed}_{train_seed}`.
  - The spindle model coefficients used for each one of the coefficient seeds are obtained by using `sampled_coefficients_{i_a,ii}_5_{coef_seed}.csv` and `coefficients.csv` in `data/extended_spindle_coefficients/{i_a,ii}/linear/`. 
  The datasets to train, test and evaluate the effect of vibrations on these models are `optimized_linear_extended_{coef_seed}_5_5_{dataset_name}.hdf5` where `coef_seed` is the same number as in the model path and `dataset_name` is as defined in the section above

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@article{perezrotondo2025illusions,
  title={Deep-learning models of the ascending proprioceptive pathway are subject to illusions},
  author={Perez Rotondo, Adriana and Simos, Merkourios and David, Florian and Pigeon, Sebastian and Blanke, Olaf and Mathis, Alexander},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.03.15.643457}
}
```

## Acknowledgements

Code for this project is partly based on two previous works in the lab: [DeepDraw](https://github.com/amathislab/DeepDraw), and [Task-driven-Proprioception](https://github.com/amathislab/Task-driven-Proprioception). We thank the authors of these works for their contributions, and we encourage you to check them out!

We thank Alberto Chiappa and Andy Bonnetto for their helpful technical input. We thank Michael Dimitriou, Anne Kavounoudias, Bianca Ziliotto and Alessandro Marin Vargas for discussions.

This project is funded by Swiss SNF grant (310030_212516), and EPFL's Excellence Research Internship Program and University of Toronto's ESROP-ExOp for S.P.

## License

This project is licensed under the [CC-BY-NC 4.0 International license](https://creativecommons.org/licenses/by-nc/4.0/).
