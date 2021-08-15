
import { io, InferenceModel, ModelPredictConfig, NamedTensorMap, Tensor, browserBlobFileRequest } from "@tensorflow/tfjs-core";

import * as tensorflow from '../data/compiled_api';
import { NamedTensorsMap, TensorInfo } from '../data/types';
import { OperationMapper } from '../operations/operation_mapper';

import { GraphExecutor } from '../executor/graph_executor';
import { ResourceManager } from '../executor/resource_manager';

export class GraphModel implements InferenceModel {
  // @ts-ignore
  private executor: GraphExecutor;
  private version = 'n/a';
  // @ts-ignore
  private handler: io.IOHandler;
  // @ts-ignore
  private artifacts: io.ModelArtifacts;
  // @ts-ignore
  private initializer: GraphExecutor;
  private resourceManager: ResourceManager;
  // @ts-ignore
  private signature: tensorflow.ISignatureDef;

  get modelVersion(): string {
    return this.version;
  }

  get inputNodes(): string[] {
    return this.executor.inputNodes;
  }

  get outputNodes(): string[] {
    return this.executor.outputNodes;
  }

  get inputs(): TensorInfo[] {
    return this.executor.inputs;
  }

  get outputs(): TensorInfo[] {
    return this.executor.outputs;
  }

  get weights(): NamedTensorsMap {
    return this.executor.weightMap;
  }

  get metadata(): {} {
    // @ts-ignore
    return this.artifacts.userDefinedMetadata;
  }

  get modelSignature(): {} {
    return this.signature;
  }

  constructor(
    private jsonPath: string,
    private weightsPath: string,
    private loadOptions: io.LoadOptions = {},
  ) {
    if (loadOptions == null) {
      loadOptions = {};
    }
    this.resourceManager = new ResourceManager();
  }

  private findIOHandler() {
    const jsonPath = this.jsonPath;
    const weigtsPath = this.weightsPath;
    this.handler = browserBlobFileRequest(jsonPath, weigtsPath);
  }

  async load(): Promise<boolean> {
    this.findIOHandler();
    if (this.handler.load == null) {
      throw new Error(
          'Cannot proceed with model loading because the IOHandler provided ' +
          'does not have the `load` method implemented.');
    }
    const artifacts = await this.handler.load();

    return this.loadSync(artifacts);
  }

  loadSync(
    artifacts: io.ModelArtifacts
  ): boolean  {
    this.artifacts = artifacts;
    const graph = this.artifacts.modelTopology as tensorflow.IGraphDef;

    let signature: tensorflow.ISignatureDef;
    if (this.artifacts.userDefinedMetadata != null &&
        this.artifacts.userDefinedMetadata.signature != null) {
      signature = // tslint:disable-next-line:no-any
          (this.artifacts.userDefinedMetadata as any).signature as
          tensorflow.ISignatureDef;
    } else {
      signature = this.artifacts.signature as tensorflow.ISignatureDef;
    }
    this.signature = signature;

    this.version = `${graph.versions?.producer}.${graph.versions?.minConsumer}`;
    const weightMap =
        io.decodeWeights(
          this.artifacts.weightData as ArrayBuffer,
          this.artifacts.weightSpecs as io.WeightsManifestEntry[]);
    this.executor = new GraphExecutor(
      OperationMapper.Instance.transformGraph(graph, this.signature));
    this.executor.weightMap = this.convertTensorMapToTensorsMap(weightMap);
    this.executor.resourceManager = this.resourceManager;

    if (artifacts.modelInitializer != null &&
      (artifacts.modelInitializer as tensorflow.IGraphDef).node != null) {
      const initializer =
          OperationMapper.Instance.transformGraph(artifacts.modelInitializer);
      this.initializer = new GraphExecutor(initializer);
      this.initializer.weightMap = this.executor.weightMap;
      this.initializer.resourceManager = this.resourceManager;
      this.initializer.executeAsync({}, []);
    }

    return true;
  }

  predict(inputs: Tensor|Tensor[]|NamedTensorMap, config?: ModelPredictConfig):
      Tensor|Tensor[]|NamedTensorMap {
    return this.execute(inputs, this.outputNodes);
  }

  private normalizeInputs(inputs: Tensor|Tensor[]|
                          NamedTensorMap): NamedTensorMap {
    if (!(inputs instanceof Tensor) && !Array.isArray(inputs)) {
      // The input is already a NamedTensorMap.
      return inputs;
    }
    inputs = Array.isArray(inputs) ? inputs : [inputs];
    if (inputs.length !== this.inputNodes.length) {
      throw new Error(
          'Input tensor count mismatch,' +
          `the graph model has ${this.inputNodes.length} placeholders, ` +
          `while there are ${inputs.length} input tensors.`);
    }
    return this.inputNodes.reduce((map, inputName, i) => {
      map[inputName] = (inputs as Tensor[])[i];
      return map;
    }, {} as NamedTensorMap);
  }

  private normalizeOutputs(outputs: string|string[]): string[] {
    outputs = outputs || this.outputNodes;
    return !Array.isArray(outputs) ? [outputs] : outputs;
  }

  execute(inputs: Tensor|Tensor[]|NamedTensorMap, outputs?: string|string[]):
      Tensor|Tensor[] {
    inputs = this.normalizeInputs(inputs);
    outputs = this.normalizeOutputs(outputs as string | string[]);
    const result = this.executor.execute(inputs, outputs);
    return result.length > 1 ? result : result[0];
  }

  async executeAsync(
      inputs: Tensor|Tensor[]|NamedTensorMap,
      outputs?: string|string[]): Promise<Tensor|Tensor[]> {
    inputs = this.normalizeInputs(inputs);
    outputs = this.normalizeOutputs(outputs as string | string[]);
    const result = await this.executor.executeAsync(inputs, outputs);
    return result.length > 1 ? result : result[0];
  }

  private convertTensorMapToTensorsMap(map: NamedTensorMap): NamedTensorsMap {
    return Object.keys(map).reduce((newMap: NamedTensorsMap, key) => {
      newMap[key] = [map[key]];
      return newMap;
    }, {});
  }

  dispose() {
    this.executor.dispose();

    if (this.initializer) {
      this.initializer.dispose();
    }

    this.resourceManager.dispose();
  }
}

export const loadGraphModel = async (
  jsonPath: string,
  weightsPath: string,
  options: io.LoadOptions = {},
): Promise<GraphModel> => {
  if (jsonPath == null || weightsPath == null) {
    throw new Error(
      'jsonPath or weightsPath in loadGraphModel() cannot be null. Please provide a url ' +
      'or an IOHandler that loads the model');
  }
  if (options == null) {
    options = {};
  }

  const model = new GraphModel(jsonPath, weightsPath);
  await model.load();
  return model;
};
