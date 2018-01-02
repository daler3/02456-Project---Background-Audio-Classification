from preprocessor import preprocessor

train_dirs = []
save_dir = 'E:\\Deep Learning Datasets\\UrbanSound8K\\extracted_short_60_overlap_9010'
for i in range(1,11):
    train_dirs.append('folder{0}_overlap_diffvol'.format(i))

pp = preprocessor(parent_dir='E:\\Deep Learning Datasets\\UrbanSound8K\\audio_overlap_diff_volumes')
print("Loading the data...")
pp.save_fts_lbs(train_dirs=train_dirs, save_path=save_dir, segment_size=20480, overlap=0.5, bands=60, frames=41)
