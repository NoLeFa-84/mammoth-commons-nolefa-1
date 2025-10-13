# data loaders
from mai_bias.catalogue.dataset_loaders.data_csv_rankings import data_csv_rankings
from mai_bias.catalogue.dataset_loaders.data_researchers import data_researchers
from mai_bias.catalogue.dataset_loaders.custom_csv import data_custom_csv
from mai_bias.catalogue.dataset_loaders.auto_csv import data_auto_csv
from mai_bias.catalogue.dataset_loaders.graph import data_graph
from mai_bias.catalogue.dataset_loaders.images import data_images
from mai_bias.catalogue.dataset_loaders.image_pairs import data_image_pairs
from mai_bias.catalogue.dataset_loaders.uci_csv import data_uci
from mai_bias.catalogue.dataset_loaders.data_any import data_read_any
from mai_bias.catalogue.dataset_loaders.free_text import data_free_text

# model loaders
from mai_bias.catalogue.model_loaders.no_model import no_model
from mai_bias.catalogue.model_loaders.manual_predictor import model_manual_predictor
from mai_bias.catalogue.model_loaders.trivial_predictor import model_trivial_predictor
from mai_bias.catalogue.model_loaders.onnx import model_onnx
from mai_bias.catalogue.model_loaders.onnx_ensemble import model_onnx_ensemble
from mai_bias.catalogue.model_loaders.pytorch import model_torch
from mai_bias.catalogue.model_loaders.pytorch2onnx import model_torch2onnx
from mai_bias.catalogue.model_loaders.fair_node_ranking import model_fair_node_ranking
from mai_bias.catalogue.model_loaders.compute_researcher_ranking import (
    model_mitigation_ranking,
    model_fair_ranking,
    model_hyperfair_ranking,
)
from mai_bias.catalogue.model_loaders.ollama import ollama_model

# metrics
from mai_bias.catalogue.metrics.model_card import model_card
from mai_bias.catalogue.metrics.specific_concerns import specific_concerns
from mai_bias.catalogue.metrics.interactive_report import interactive_report
from mai_bias.catalogue.metrics.sklearn_audit import sklearn_audit
from mai_bias.catalogue.metrics.sklearn_visual_analysis import sklearn_visual_analysis
from mai_bias.catalogue.metrics.image_bias_analysis import image_bias_analysis
from mai_bias.catalogue.metrics.xai_analysis import facex_regions
from mai_bias.catalogue.metrics.xai_analysis_embeddings import facex_embeddings
from mai_bias.catalogue.metrics.ranking_fairness import exposure_distance_comparison
from mai_bias.catalogue.metrics.multi_objective_report import multi_objective_report
from mai_bias.catalogue.metrics.viz_fairness_plots import viz_fairness_plots
from mai_bias.catalogue.metrics.viz_fairness_report import viz_fairness_report
from mai_bias.catalogue.metrics.optimal_transport import optimal_transport
from mai_bias.catalogue.metrics.bias_scan import bias_scan
from mai_bias.catalogue.metrics.croissant import croissant
from mai_bias.catalogue.metrics.aif360 import aif360
from mai_bias.catalogue.metrics.augmentation_report import (
    augmentation_report,
)
from mai_bias.catalogue.metrics.text_dbias import text_debias
from mai_bias.catalogue.metrics.self_critic import llm_audit

from mai_bias.backend.registry import Registry

registry = Registry()

registry.data(data_auto_csv)
registry.data(data_uci)
registry.data(data_custom_csv)
registry.data(data_csv_rankings)
registry.data(data_researchers)
registry.data(data_graph)
registry.data(data_images)
registry.data(data_image_pairs)
registry.data(data_free_text)
registry.data(data_read_any)

registry.model(
    no_model,
    compatible=[
        data_auto_csv,
        data_custom_csv,
        data_uci,
        data_read_any,
        data_images,
        data_image_pairs,
        data_free_text,
    ],
)
registry.model(
    ollama_model,
    compatible=[
        data_free_text,
    ],
)
registry.model(
    model_manual_predictor,
    compatible=[
        data_auto_csv,
        data_custom_csv,
        data_uci,
        data_read_any,
        data_images,
        data_image_pairs,
    ],
)
registry.model(
    model_trivial_predictor,
    compatible=[
        data_auto_csv,
        data_custom_csv,
        data_uci,
        data_read_any,
        data_images,
        data_image_pairs,
    ],
)
registry.model(
    model_onnx, compatible=[data_auto_csv, data_custom_csv, data_uci, data_read_any]
)
registry.model(
    model_onnx_ensemble,
    compatible=[data_auto_csv, data_custom_csv, data_uci, data_read_any],
)
registry.model(model_torch, compatible=[data_images, data_image_pairs])
registry.model(model_torch2onnx, compatible=[data_images, data_image_pairs])
registry.model(model_fair_node_ranking, compatible=[data_graph])
registry.model(model_mitigation_ranking, compatible=[data_researchers])
registry.model(model_hyperfair_ranking, compatible=[data_researchers])
registry.model(model_fair_ranking, compatible=[data_researchers])

registry.analysis(model_card)
registry.analysis(specific_concerns)
registry.analysis(interactive_report)
registry.analysis(sklearn_audit)
registry.analysis(sklearn_visual_analysis)
registry.analysis(aif360)
registry.analysis(optimal_transport)
registry.analysis(bias_scan)
registry.analysis(image_bias_analysis)
registry.analysis(facex_regions)
registry.analysis(facex_embeddings)
registry.analysis(multi_objective_report)
registry.analysis(viz_fairness_plots)
registry.analysis(viz_fairness_report)
registry.analysis(croissant)
registry.analysis(
    exposure_distance_comparison,
    compatible=[model_mitigation_ranking, model_fair_ranking, model_hyperfair_ranking],
)
registry.analysis(augmentation_report)
registry.analysis(text_debias)
registry.analysis(llm_audit)
