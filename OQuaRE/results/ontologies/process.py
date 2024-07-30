import xml.etree.ElementTree as ET
import argparse
import pandas as pd
from collections import OrderedDict
import os.path as osp

def find_metrics(root, main_tag, main_attr, submetrics):
    main_element = root.find(f".//{main_tag}")
    if main_element is not None:
        main_value = main_element.attrib.get(main_attr, 'N/A')
        submetric_values = {sub: main_element.find(sub).text if main_element.find(sub) is not None else 'N/A' for sub in submetrics}
        return main_value, submetric_values
    return 'N/A', {sub: 'N/A' for sub in submetrics}

def metrics_to_latex(parsed_metrics):
    latex_str = r"\begin{table}[h!]" + "\n" + r"\begin{tabular}{ll}" + "\n" + r"\hline"
    
    for category, metrics in parsed_metrics.items():
        main_value = float(metrics['main'])
        latex_str += f"\n\t{main_value:.2f} \\\\"
        for metric, value in metrics.items():
            if metric != 'main':
                metric_formatted = ' '.join(metric.split()).capitalize()
                submetric_value = float(value) if value != 'N/A' else value
                latex_str += f"\n\t{submetric_value:.2f} \\\\" if value != 'N/A' else f"\n{metric_formatted} & {value} \\\\"
        latex_str += "\n\\hline"
    
    latex_str += "\n\\end{tabular}" + "\n\\end{table}"
    return latex_str

def metrics_to_pandas(parsed_metrics):
    data = OrderedDict()
    for category, metrics in parsed_metrics.items():
        for metric, value in metrics.items():
            if metric != 'main':
                data[metric] = [float(value)]
            else:
                data[category] = [float(value)]

    return pd.DataFrame.from_dict(data)

def process_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    metrics = {
        'structural': {
            'main_tag': 'oquareModelStructural',
            'main_attr': 'structural',
            'submetrics': ['cohesion', 'consistency', 'formalRelationSupport', 'formalisation', 'redundancy', 'tagledness']
        },
        'functionaladequacy': {
            'main_tag': 'oquareModelFunctionalAdequacy',
            'main_attr': 'functionalAdequacy',
            'submetrics': [
                'clusteringAndSimilarity', 'consistentSearchAndQuery', 'controlledVocabulary', 'guidanceAndDecisionTrees',
                'indexingAndLinking', 'infering', 'knowledgeAcquisition', 'knowledgeReuse', 'referenceOntology',
                'resultsRepresentation', 'schemaAndValueReconciliation', 'textAnalysis'
            ]
        },
        'maintainability': {
            'main_tag': 'oquareModelMaintainability',
            'main_attr': 'maintainability',
            'submetrics': ['analysability', 'changeability', 'modificationStability', 'modularity', 'reusability', 'testeability']
        },
        'total': {
            'main_tag': 'oquareModel',
            'main_attr': 'oquareValue',
            'submetrics': []
        }
    }

    parsed_metrics = {}
    for key, value in metrics.items():
        main_value, submetric_values = find_metrics(root, value['main_tag'], value['main_attr'], value['submetrics'])
        parsed_metrics[key] = {'main': main_value, **submetric_values}

    # latex_table = metrics_to_latex(parsed_metrics)
    results_df = metrics_to_pandas(parsed_metrics)
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Process XML file to extract metrics and generate LaTeX table.')
    parser.add_argument('xml_file', type=str, help='Path to the XML file to process')
    args = parser.parse_args()

    file_path = args.xml_file

    results_df = process_xml(file_path)
    filename = osp.splitext(osp.basename(file_path))[0]
    dirname = osp.dirname(file_path)
    out_path = osp.join(dirname, filename + '.csv')
    results_df.to_csv(out_path, index=False)
    print(f'Saved results to {out_path}')

if __name__ == "__main__":
    main()


