from marked_lineart_vec.models.base import BaseVAE
from marked_lineart_vec.models.line_identification import LineIdentificationModel
from marked_lineart_vec.models.marked_reconstruction import MarkedReconstructionModel
from marked_lineart_vec.models.vanilla_vae import VanillaVAE
from marked_lineart_vec.models.vector_vae import VectorVAE
from marked_lineart_vec.models.vector_vae_nlayers import VectorVAEnLayers
from marked_lineart_vec.models.iterative import IterativeModel


vae_models = {
    'VanillaVAE': VanillaVAE,
              'VectorVAE': VectorVAE,
              'VectorVAEnLayers': VectorVAEnLayers,
              'IterativeModel': IterativeModel,
"MarkedReconstructionModel": MarkedReconstructionModel,
"LineIdentificationModel": LineIdentificationModel}
