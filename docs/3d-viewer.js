// srcs: https://github.com/elliotwaite/pytorch-to-javascript-with-onnx-js/blob/master/full_demo/script.js

const CANVAS_SIZE = 280;
const CANVAS_SCALE = 0.5;

import * as THREE from './threejs-light/three.module.js'
import {OrbitControls} from './threejs-light/OrbitControls.js'
import { GUI } from './threejs-light/lil-gui.module.min.js';

let mesh

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 50, window.innerWidth / window.innerHeight, 0.001, 1 );

const renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );

const container = document.getElementById( 'renderer' );
container.appendChild( renderer.domElement );

function render() {renderer.render( scene, camera );}

// Control camera via mouse
const controls = new OrbitControls( camera, renderer.domElement );
controls.target.set( 0, 0, 0 );
controls.update();
controls.enablePan = false;
controls.enableDamping = true;

const light = new THREE.HemisphereLight( 0xffffff, 0x080808, 1. );
const light2 = new THREE.HemisphereLight( 0xffffff, 0x080808, 0.4 );
light.position.set( 0.1, 0, 1 );
light2.position.set( 0, 0, -1 );
scene.add( light );
scene.add( light2 )

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
const settings = {'show_template': false, 'footedness': 'Left', 'reset_model': reset_model, cam_dist: 0.3}
var latent_means = {}
var latent_vecs = {}
var latent_stds = {}
const model_params = {scale_X: 1, scale_Y: 1, scale_Z: 1};

// Save examples of latent variables
const pose_targets = {'T-Pose':0, 'Toe Extension': 0, 'Dorsiflex':0}
var poses = {}

function read_json(jsonText){
	const data = JSON.parse(jsonText)
	for (const latent of latent_keys){
		latent_means[latent] = data[latent]['mean']
		latent_vecs[latent] = data[latent]["V"]
		latent_stds[latent] = data[latent]['stddev']
	}

	for (const pose_example of data['pose_examples']){
		var pose_str = pose_example['pose'].join(" ")
		if (Object.keys(pose_targets).includes(pose_str)){
			if (pose_targets[pose_str] === 0){
				pose_targets[pose_str] = 1 // Mark as found
				poses[pose_str] = pose_example['pca']
			}
		}
	}

	json_loaded = true
}

function reset_model(){
	for (const k of Object.keys(model_params)) {
		model_params[k] = 0
	}
	updateModel(model_params)
}

function set_footedness(footedness){
	if (footedness === 'Left'){
		mesh.scale.y = 1;
	}
	else {mesh.scale.y = -1}
}

function set_example_pose(pose_str){
	// Given a pose string of an example pose, set this to be the active pose
	const pca_params = poses[pose_str]
	for (var i=1; i < 1 + pca_components; i++) {
		model_params['pose' + i] = pca_params[i-1]
	}
	updateModel(model_params)
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
	camera.position.z = settings['cam_dist'];

	// folders['scale'] = gui.addFolder('Scale');
	folders['shape'] = gui.addFolder('Shape');
	folders['pose'] = gui.addFolder('Pose');
	folders['tex'] = gui.addFolder('Texture');
	folders['settings'] = gui.addFolder('Viewing Settings');

	// // folders['scale'].add(params, 'scale_X', 0.6, 1.4).name('Scale X').onChange(function (value) {
	// 		for (let j = 0; j < position.count; j++) {position.setX(j, template_vertices.getX(j) * value)}}).listen(reset_model)
	//
	// folders['scale'].add(params, 'scale_Y', 0.6, 1.4).name('Scale Y').onChange(function (value) {
	// 		for (let j = 0; j < position.count; j++) {position.setY(j, template_vertices.getY(j) * value)}}).listen(reset_model)
	//
	// folders['scale'].add(params, 'scale_Z', 0.6, 1.4).name('Scale Z').onChange(function (value) {
	// 		for (let j = 0; j < position.count; j++) {position.setZ(j, template_vertices.getZ(j) * value)}}).listen(reset_model)

	folders['settings'].add(settings, 'cam_dist', 0.2, 0.5).name('View distance').onChange(function(value){
		camera.position.z = value
	})

	folders['settings'].add(settings, 'show_template').name('Show template').onChange(function(value){updateModel(model_params)})
	folders['settings'].add(settings, 'footedness', ['Left', 'Right']).name('Foot').onChange(set_footedness)
	folders['settings'].add(settings, 'reset_model').name('Reset model')

	document.getElementById('Num verts').innerText = position.count
	document.getElementById('Num faces').innerText = obj_data.faces.length / 3

	template_vertices = position.clone()
	animate()

}


function setupModel(){

	const f = function (value) {updateModel(model_params)}
	for (var i=1; i<1 + pca_components; i++) {
		for (const k of ['shape', 'tex', 'pose']) {
			model_params[k + i] = 0
			folders[k].add(model_params, k + i, -3, 3).name('PC ' + i).onChange(f).listen(reset_model);
		}
	}

	var pose_options = {}
	folders['selected_pose'] = folders['pose'].addFolder('Select Pose')
	for (const example_pose of Object.keys(pose_targets)){
		pose_options[example_pose] = function() {set_example_pose(example_pose)}
		folders['selected_pose'].add(pose_options, example_pose).name(example_pose)
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
	const points = new onnx.Tensor(template_vertices.array, "float32", [N, 3])

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

	var startTime = performance.now()
	const outputMap = await sess.run([points, vecs['shape'], vecs['tex'], vecs['pose']])
	var endTime = performance.now()
	document.getElementById('Net eval time').innerText = (endTime - startTime).toFixed(0) + " ms"


	var disp = outputMap.get('disp')
	var col_val = outputMap.get('col')

    for (let i=0; i<N; i++) {
		if (settings['show_template']){
			// Show original template vertices
			position.setXYZ(i, template_vertices.getX(i),
				template_vertices.getY(i),
				template_vertices.getZ(i))
			colours.setXYZ(i, 0.5, 0.5, 0.5)
		}
		else {
			// Colour and displace vertex
			position.setXYZ(i, template_vertices.getX(i) + disp.get(i, 0),
				template_vertices.getY(i) + disp.get(i, 1),
				template_vertices.getZ(i) + disp.get(i, 2))
			colours.setXYZ(i, col_val.get(i, 0), col_val.get(i, 1), col_val.get(i, 2));
		}
    }
}

loadingModelPromise.then(() => {
	if (!obj_loaded){
    	console.log("OBJ not loaded - waiting...")}
	while(!obj_loaded){}
	if (!json_loaded){
    	console.log("JSON not loaded - waiting...")}
	while(!json_loaded){}

	console.log("OBJ, JSON, ONNX loaded.")
	setupModel()

})

window.addEventListener( 'resize', onWindowResize, false );

function onWindowResize(){

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize( window.innerWidth, window.innerHeight );

}