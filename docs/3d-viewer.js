// srcs: https://github.com/elliotwaite/pytorch-to-javascript-with-onnx-js/blob/master/full_demo/script.js

const CANVAS_SIZE = 280;
const CANVAS_SCALE = 0.5;

import * as THREE from './threejs-light/three.module.js'
import {OrbitControls} from './threejs-light/OrbitControls.js'
import { GUI } from './threejs-light/lil-gui.module.min.js';

let mesh

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );

const renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );
document.body.appendChild( renderer.domElement );

function render() {renderer.render( scene, camera );}

// Control camera via mouse
const controls = new OrbitControls( camera, renderer.domElement );
controls.target.set( 0, 0, 0 );
controls.update();
controls.enablePan = false;
controls.enableDamping = true;

const light = new THREE.HemisphereLight( 0xffffff, 0x080808, 1. );
light.position.set( 0.1, 0, 1 );
scene.add( light );

let geometry

// flags to validate loading
var obj_loaded = false
var json_loaded = false

fetch('./find-demo-model/template.obj').then(response => response.text()).then(text => read_obj(text)).then(setupScene)
fetch('./find-demo-model/latents.json').then(response => response.text()).then(text => read_json(text))

let position, colours
let template_vertices
let obj_data
var gui = new GUI();
var folders = {}

function duplicate(array, N){
	// Stack array N times,
	//eg [2,3], N=5 -> [2,3,2,3,2,3,2,3,2,3]
	return Array(N).fill(array).flat()
}

function read_obj(objText) {
	// Read OBJ to vertices and faces (does NOT support OBJ files with vertex normals/texture coordinates currently)
	obj_data = {};
	const vertices = []
	const faces = []
	var vertexMatches = objText.match(/^v( -?\d+(\.\d+)?){3}$/gm);
	var faceMatches = objText.match(/^f( -?\d+(\.\d+)?){3}$/gm);
	if (vertexMatches)
	{
		vertexMatches.map(function(vertex)
		{
			var v = vertex.split(" ");
			v.shift();
			vertices.push.apply(vertices, v.map(parseFloat))
		})
		obj_data.vertices = new THREE.Float32BufferAttribute( new Float32Array(vertices ), 3);
	}

	if (faceMatches){
		faceMatches.map(function(face)
		{
			var idxs = face.split(" ");
			idxs.shift();
			faces.push.apply(faces, idxs.map(num => parseInt(num, 10) - 1))

		})
		obj_data.faces = faces;
	}

}

const pca_components = 3
const latent_keys = ['shape', 'pose', 'tex']
var latent_means = {}
var latent_vecs = {}
var latent_stds = {}

function read_json(jsonText){
	const data = JSON.parse(jsonText)
	for (const latent of latent_keys){
		latent_means[latent] = data[latent]['mean']
		latent_vecs[latent] = data[latent]["V"]
		latent_stds[latent] = data[latent]['stddev']
	}
}

function setupScene() {

	obj_loaded = true

	geometry = new THREE.BufferGeometry()
	geometry.setIndex(obj_data.faces)
	geometry.setAttribute("position", obj_data.vertices)
	geometry.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(obj_data.vertices.count * 3).fill(0.5), 3))
	geometry.computeVertexNormals()

	position = geometry.getAttribute('position');
	colours = geometry.getAttribute('color')

	const material = new THREE.MeshPhongMaterial( {
					side: THREE.DoubleSide,
					vertexColors: true
				} );

	mesh = new THREE.Mesh(geometry, material);
	scene.add(mesh);

	// gui
	const params = {X: 1, Y: 1, Z: 1, cam_dist: 0.3};

	camera.position.z = params['cam_dist'];

	folders['scale'] = gui.addFolder('Scale');
	folders['shape'] = gui.addFolder('Shape');
	folders['pose'] = gui.addFolder('Pose');
	folders['tex'] = gui.addFolder('Texture');
	folders['settings'] = gui.addFolder('Viewing Settings');

	folders['scale'].add(params, 'X', 0.5, 2).name('Scale X').onChange(function (value) {
			for (let j = 0; j < position.count; j++) {position.setX(j, template_vertices.getX(j) * value)}})

	folders['scale'].add(params, 'Y', 0.5, 2).name('Scale Y').onChange(function (value) {
			for (let j = 0; j < position.count; j++) {position.setY(j, template_vertices.getY(j) * value)}})

	folders['scale'].add(params, 'Z', 0.5, 2).name('Scale Z').onChange(function (value) {
			for (let j = 0; j < position.count; j++) {position.setZ(j, template_vertices.getZ(j) * value)}})

	folders['settings'].add(params, 'cam_dist', 0.2, 0.5).name('View distance').onChange(function(value){
		camera.position.z = value
	})

	template_vertices = position.clone()
	animate()

}


function setupModel(){
	const model_params = {};

	const f = function (value) {updateModel(model_params)}
	for (var i=1; i<4; i++) {
		for (const k of ['shape', 'tex', 'pose']) {
			model_params[k + i] = 0
			folders[k].add(model_params, k + i, -3, 3).name('PC ' + i).onChange(f);
		}
	}

	updateModel(model_params)
}

const animate = function () {
	requestAnimationFrame( animate );

	position.needsUpdate = true;
	colours.needsUpdate = true
	controls.update();
	renderer.render( scene, camera );
};

// Load our model.
const sess = new onnx.InferenceSession();
const loadingModelPromise = sess.loadModel("./find-demo-model/model.onnx");

async function updateModel(model_params) {
	// Update the FIND model with the current parameters

	const N = position.count
	const points = new onnx.Tensor(template_vertices.array.slice(0, 3*N), "float32", [N, 3])

	var vecs = {}
	for (var k of latent_keys) {

		var arr = latent_means[k]
		for (var i = 0; i < pca_components; i++) {
			var comp = latent_vecs[k][i]
			var mag = model_params[k + (i + 1)] * latent_stds[k][i]
			arr = arr.map((a, j) => a + comp[j] * mag);
		}

		vecs[k] = new onnx.Tensor(duplicate(arr, N), "float32", [N, 100])

	}
	// console.log(texvec_arr.reduce((partialSum, a) => partialSum + a, 0))

	var startTime = performance.now()
	const outputMap = await sess.run([points, vecs['shape'], vecs['tex'], vecs['pose']])
	var endTime = performance.now()
	console.log(`Network eval for N=${N} points took ${endTime - startTime} milliseconds`)

	var disp = outputMap.get('disp')
	var col_val = outputMap.get('col')

    for (let i=0; i<N; i++) {
		// Colour vertex
		position.setXYZ(i, template_vertices.getX(i) + disp.get(i,0),
			template_vertices.getY(i) + disp.get(i,1),
			template_vertices.getZ(i) + disp.get(i,2))
		colours.setXYZ(i, col_val.get(i,0), col_val.get(i,1), col_val.get(i,2) );

    }
}

loadingModelPromise.then(() => {
	console.log("Loaded ONNX model.")
	if (!obj_loaded){
    	console.log("OBJ not loaded - waiting...")}
	while(!obj_loaded){}
	console.log("OBJ loaded!")
	setupModel()

})