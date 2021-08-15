import { getModelArtifactsForJSONCustom } from './io_utils';
import { IOHandler, ModelJSON, ModelArtifacts } from './types';

import * as localForage from 'localforage';

/**
 * 自定义加载缓存在内存中的类
 *
 * @export
 * @class BrowserBlobFile
 * @implements {io.IOHandler}
 */
 export class BrowserBlobFile implements IOHandler {
  protected readonly jsonPath: string;
  protected readonly weightsPath: string;

  constructor(jsonPath: string, weightPath: string) {
    this.jsonPath = jsonPath;
    this.weightsPath = weightPath;
  }

  async load(): Promise<ModelArtifacts> {
    // 从 IndexedDB or localStorage 中加载缓存的模型文件和权重文件
    let jsonFile = await localForage.getItem(this.jsonPath) as Uint8Array;
    const weight = await localForage.getItem(this.weightsPath) as Uint8Array;
    const str = jsonFile.reduce((total: string, byte: number) => {
      return total =  total + String.fromCharCode(byte);
    }, '');
    const modelJSON = JSON.parse(str) as ModelJSON;

    const weightsSpecs = [] as Array<any>;
    modelJSON.weightsManifest.forEach((item) => {
      weightsSpecs.push(...item.weights);
    });

    // const weights = [] as Array<Uint8Array>;
    // let totalLength = 0;
    // for (let path of this.weightsPath) {
    //   let weight = await localForage.getItem(path) as Uint8Array;
    //   totalLength += weight.length;
    //   weights.push(weight);
    // }

    // const weightsUint8 = new Uint8Array(totalLength);
    // let offset = 0
    // weights.forEach((item) => {
    //   weightsUint8.set(item, offset);
    //   offset += item.length;
    // });

    return getModelArtifactsForJSONCustom(modelJSON, weightsSpecs, weight.buffer);
  }
}

export const browserBlobFileRequest = (
  jsonPath: string,
  weightsPath: string,
): IOHandler => {
  return new BrowserBlobFile(jsonPath, weightsPath);
};
