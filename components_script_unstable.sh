echo "Building unstable components"

pip install --upgrade -r requirements_gh_build.txt
pip install -e .


kfp component build . --component-filepattern mai_bias/catalogue/metrics/aif360_metrics.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/metrics/bias_scan.py
docker system prune -a --force --volumes
kfp component build . --component-filepattern mai_bias/catalogue/metrics/optimal_transport.py
docker system prune -a --force --volumes

mkdir yamls
mkdir yamls/data
mkdir yamls/meta

cp mai_bias/catalogue/dataset_loaders/component_metadata/* yamls/meta/
cp mai_bias/catalogue/model_loaders/component_metadata/* yamls/meta/
cp mai_bias/catalogue/metrics/component_metadata/* yamls/meta/
cp component_metadata/* yamls/data/

echo "Completed building unstable components"

