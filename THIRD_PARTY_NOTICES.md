# Third-Party Data Notices

This repository contains processed versions of third-party NLOS measurements.
The MIT license in `LICENSE` and the CC BY 4.0 terms in `DATA_LICENSE` do not
replace the upstream terms for the files listed here. Users are responsible for
complying with both this notice and the applicable upstream terms.

## Teaser and Statue measurements

### Covered files

- `real_data/real_trans_dataset_teaser_128pattern.mat`
- `real_data/real_trans_dataset_teaser_random3000.mat`
- `real_data/real_trans_dataset_statue_center256_1.0m.mat`
- `real_data/real_trans_dataset_statue_center256_0.5m.mat`
- `real_output/teaser_128pattern/`
- `real_output/teaser_random3000/`
- `real_output/statue_center256_1.0m_2/`
- `real_output/statue_center256_0.5m_10/`

The `real_data` files are processed and subsampled forms of the Teaser and
Statue measurements distributed with the nlos-fk project. The listed
`real_output` directories contain 3D-GTR models, synthesized transients, and
reconstructions derived from those measurements.

### Source and permission terms

- Project: [Wave-Based Non-Line-of-Sight Imaging using Fast f-k Migration](https://www.computationalimaging.org/publications/nlos-fk/)
- Repository: [computational-imaging/nlos-fk](https://github.com/computational-imaging/nlos-fk)
- Controlling upstream license: [nlos-fk LICENSE](https://github.com/computational-imaging/nlos-fk/blob/master/LICENSE)
- Upstream copyright notice: Copyright (c) 2018, Stanford University. All
  rights reserved.

The upstream custom license permits redistribution and use, with or without
modification, for academic and other non-commercial purposes. Its conditions
include retaining the Stanford copyright notice, license conditions, and
disclaimer; reproducing them with binary or modified distributions; avoiding
Stanford or contributor endorsement; and making publicly redistributed modified
source freely accessible or available at no charge. The upstream software and
data are provided without warranty. Consult the linked license text for the
complete controlling terms.

The 3D-GTR authors have separately confirmed permission to redistribute the
processed files included here. This notice does not grant commercial-use rights
to the upstream measurements. Anyone needing rights beyond the upstream terms
or the authors' existing permission should contact the original rights holder.

### Required acknowledgment

The upstream project asks dataset users to acknowledge the work by citing the
following publications:

1. Matthew O'Toole, Felix Heide, David B. Lindell, Kai Zang, Steven Diamond,
   and Gordon Wetzstein. "Reconstructing Transient Images from Single-Photon
   Sensors." IEEE Conference on Computer Vision and Pattern Recognition, 2017.
2. Matthew O'Toole, David B. Lindell, and Gordon Wetzstein. "Confocal
   Non-Line-of-Sight Imaging Based on the Light-Cone Transform." Nature 555,
   338-341, 2018.
3. Felix Heide, Matthew O'Toole, Kai Zang, David B. Lindell, Steven Diamond,
   and Gordon Wetzstein. "Non-Line-of-Sight Imaging with Partial Occluders and
   Surface Normals." ACM Transactions on Graphics 38(3), Article 22, 2019.
   https://doi.org/10.1145/3269977
4. David B. Lindell, Gordon Wetzstein, and Matthew O'Toole. "Wave-Based
   Non-Line-of-Sight Imaging Using Fast f-k Migration." ACM Transactions on
   Graphics 38(4), Article 116, 2019.

## Zaragoza Bunny synthetic measurements

### Covered files

- `real_data/real_trans_dataset_bunny_128pattern.mat`
- `real_data/real_trans_dataset_bunny_center256_0.5m.mat`
- `real_data/real_trans_dataset_bunny_res8.mat`
- `real_output/bunny_128pattern_3/`
- `real_output/bunny_center0.5m_2/`
- `real_output/bunny_res8_3/`

The `real_data` files are processed or subsampled forms of the Stanford Bunny
scene from the Zaragoza NLOS synthetic dataset. The listed `real_output`
directories contain 3D-GTR models, synthesized transients, and reconstructions
derived from those measurements.

### Source and permission terms

- Dataset: [Zaragoza NLOS synthetic dataset](https://graphics.unizar.es/nlos_dataset.html)
- Provider: Graphics and Imaging Lab, Universidad de Zaragoza-I3A

The Zaragoza NLOS synthetic dataset is publicly made available for research
use, and its project page requests citation of the two works below. The project
page does not display a standard license identifier or provide separate dataset
license text. This repository includes processed and subsampled Bunny
measurements for academic research and reproducibility in accordance with that
stated research purpose. The upstream data and the processed files listed here
are not covered by this repository's CC BY 4.0 data license. Users must retain
the required acknowledgments and should consult the original provider regarding
uses outside academic research, including commercial use or further
redistribution.

### Required acknowledgment

The dataset page asks users to cite both:

1. Miguel Galindo, Julio Marco, Matthew O'Toole, Gordon Wetzstein, Diego
   Gutierrez, and Adrian Jarabo. "A Dataset for Benchmarking Time-Resolved
   Non-Line-of-Sight Imaging." IEEE International Conference on Computational
   Photography, 2019. https://graphics.unizar.es/nlos
2. Adrian Jarabo, Julio Marco, Adolfo Munoz, Raul Buisan, Wojciech Jarosz, and
   Diego Gutierrez. "A Framework for Transient Rendering." ACM Transactions on
   Graphics 33(6), Article 177, 2014.

## Citation of 3D-GTR

Use of the processed datasets or derived artifacts should also cite the 3D-GTR
paper as described in `README.md` and `CITATION.cff`. Citing 3D-GTR does not
replace the upstream acknowledgments above.
