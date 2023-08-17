from maddeb.dataset_generator import loadCATSIMDataset

loadCATSIMDataset(
    train_data_dir='/sps/lsst/users/bbiswas/simulations/CATSIM_tfDataset/blended_training',
    val_data_dir='/sps/lsst/users/bbiswas/simulations/CATSIM_tfDataset/blended_validation',
    output_dir='/sps/lsst/users/bbiswas/simulations/CATSIM_tfDataset/blended_tfDataset'
)

loadCATSIMDataset(
    train_data_dir='/sps/lsst/users/bbiswas/simulations/CATSIM_tfDataset/isoled_training',
    val_data_dir='/sps/lsst/users/bbiswas/simulations/CATSIM_tfDataset/isolated_validation',
    output_dir='/sps/lsst/users/bbiswas/simulations/CATSIM_tfDataset/isolated_tfDataset'
)