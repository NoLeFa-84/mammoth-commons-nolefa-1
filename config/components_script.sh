echo "Building components"

pip install --upgrade -r config/requirements_gh_build.txt
pip install -e .

kfp component build . --component-filepattern mai_bias/catalogue/dataset_loaders/auto_csv.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/dataset_loaders/custom_csv.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/dataset_loaders/data_any.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/dataset_loaders/data_csv_rankings.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/dataset_loaders/data_researchers.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/dataset_loaders/free_text.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/dataset_loaders/graph.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/dataset_loaders/image_pairs.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/dataset_loaders/images.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/dataset_loaders/uci_csv.py
docker system prune -a --force --volumes

kfp component build . --component-filepattern mai_bias/catalogue/model_loaders/compute_rankings.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/model_loaders/compute_researcher_ranking.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/model_loaders/fair_node_ranking.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/model_loaders/manual_predictor.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/model_loaders/no_model.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/model_loaders/onnx_ensemble.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/model_loaders/ollama.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/model_loaders/onnx.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/model_loaders/pytorch.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/model_loaders/pytorch2onnx.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/model_loaders/trivial_predictor.py
docker system prune -a --force --volumes

kfp component build . --component-filepattern mai_bias/catalogue/metrics/aif360_metrics.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/metrics/augmentation_report.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/metrics/bias_scan.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/metrics/croissant.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/metrics/image_bias_analysis.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/metrics/interactive_report.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/metrics/model_card.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/metrics/multi_objective_report.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/metrics/networks_layouts.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/metrics/optimal_transport.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/metrics/ranking_fairness.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/metrics/sklearn_audit.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/metrics/specific_concerns.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/metrics/text_dbias.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/metrics/viz_fairness_plots.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/metrics/viz_fairness_report.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/metrics/xai_analysis_embeddings.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/metrics/xai_analysis.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/metrics/self_critic.py
docker system prune -a --force --volumes

mkdir yamls
mkdir yamls/data
mkdir yamls/meta

cp mai_bias/catalogue/dataset_loaders/component_metadata/* yamls/meta/
cp mai_bias/catalogue/model_loaders/component_metadata/* yamls/meta/
cp mai_bias/catalogue/metrics/component_metadata/* yamls/meta/
cp component_metadata/* yamls/data/

echo "Completed building components"

