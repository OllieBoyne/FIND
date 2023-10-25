// srcs: https://github.com/elliotwaite/pytorch-to-javascript-with-onnx-js/blob/master/full_demo/script.js

const CANVAS_SIZE = 280;
const CANVAS_SCALE = 0.5;

import * as THREE from 'https://unpkg.com/three/build/three.module.js';
// import * as THREE from './threejs-light/three.module.js'
import {OrbitControls} from './threejs-light/OrbitControls.js'
import { GUI } from './threejs-light/lil-gui.module.min.js';
import { OBJLoader } from './threejs-light/OBJLoader.js'

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
var obj_loaded = false
let object
const loader = new OBJLoader()

fetch('./find-demo-model/template.obj').then(response => response.text()).then(text => read_obj(text)).then(setupScene)


// loader.load('./find-demo-model/template.obj',
// 	function(obj) {object=obj; scene.add(obj); setupScene(obj); obj_loaded = true})
	// function ( xhr ) {console.log( ( xhr.loaded / xhr.total * 100 ) + '% loaded' );},
	// function ( error ) {console.log( 'An error happened' );})

let position, colours
let template_vertices
let obj_data
var gui = new GUI();

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
			vertices.push.apply(vertices, v.map(parseFloat).reverse())
		})
		obj_data.vertices = new THREE.Float32BufferAttribute( new Float32Array(vertices ), 3);
	}

	if (faceMatches){
		faceMatches.map(function(face)
		{
			var idxs = face.split(" ");
			idxs.shift();
			faces.push.apply(faces, idxs.map(num => parseInt(num, 10)))

		})
		obj_data.faces = faces;
	}

}

function setupScene() {

	obj_loaded = true

	// const geometry = new THREE.BufferGeometry();
	// const geometry = new THREE.BoxGeometry(0.01)
	// geometry = obj.children[0].geometry
	// geometry = BufferGeometryUtils.mergeVertices(geometry) // Merge duplicate vertices

	geometry = new THREE.BufferGeometry()
	geometry.setIndex(obj_data.faces)
	geometry.setAttribute("position", obj_data.vertices)
	geometry.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(obj_data.vertices.count * 3).fill(0.5), 3))
	geometry.computeVertexNormals()

	position = geometry.getAttribute('position');

	// geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(position.count * 3).fill(1), 3))
	// colours = geometry.getAttribute('color');
	// const material = new THREE.MeshBasicMaterial({color: 0x008888});
	// const material = new THREE.MeshBasicMaterial({ vertexColors: true })

	const material = new THREE.MeshPhongMaterial( {
					side: THREE.DoubleSide,
					vertexColors: true
				} );

	mesh = new THREE.Mesh(geometry, material);
	scene.add(mesh);

	camera.position.z = 0.3;

	// gui
	const params = {X: 1, Y: 1, Z: 1};

	var scale_folder = gui.addFolder('Scale');

	scale_folder.add(params, 'X', 0.5, 2).name('Scale X').onChange(function (value) {
			for (let j = 0; j < position.count; j++) {position.setX(j, template_vertices.getX(j) * value)}})

	scale_folder.add(params, 'Y', 0.5, 2).name('Scale Y').onChange(function (value) {
			for (let j = 0; j < position.count; j++) {position.setY(j, template_vertices.getY(j) * value)}})

	scale_folder.add(params, 'Z', 0.5, 2).name('Scale Z').onChange(function (value) {
			for (let j = 0; j < position.count; j++) {position.setZ(j, template_vertices.getZ(j) * value)}})

	template_vertices = position.clone()
	animate()

}


function setupModel(){
	const model_params = {shape1: 0};
	var f2 = gui.addFolder('Shape');
	f2.add(model_params, 'shape1', -1, 1).name('Shape 1').onChange(function (value) {updateModel(model_params)});
	updateModel(model_params)
}

const animate = function () {
	requestAnimationFrame( animate );

	position.needsUpdate = true;
	// colours.needsUpdate = true
	controls.update();
	renderer.render( scene, camera );
};

// animate();
// renderer.render( scene, camera );

// Load our model.
const sess = new onnx.InferenceSession();
const loadingModelPromise = sess.loadModel("./find-demo-model/model.onnx");

async function updateModel(model_params) {
  // Get the predictions model offsets.
  //   const points = new onnx.Tensor(new Float32Array(template_vertices.array), "float32")
	var col, face
    for (let i=0; i< 10; i++) {
        const shapevec = new onnx.Tensor(new Array(100).fill(0), "float32", [1, 100])
        const texvec = new onnx.Tensor(new Array(100).fill(0), "float32", [1, 100])
        const posevec = new onnx.Tensor(new Array(100).fill(0), "float32", [1, 100])
        const points = new onnx.Tensor([template_vertices.getX(i), template_vertices.getY(i), template_vertices.getZ(i)], "float32", [1, 3])

        const outputMap = await sess.run([points, shapevec, texvec, posevec])

        var disp = outputMap.get('disp')
        var col_val = outputMap.get('Col')

        // position.setX(i, 0)
        // position.setY(i, 0)
        // position.setZ(i, 0)

		// Colour vertex
		// col = new THREE.Color( 0xffffff );
		// col.setRGB( col_val[0], col_val[1], col_val[2] );
		// geometry.colors[i] = col; // use this array for convenience

    }

	// copy the colors to corresponding positions
	//     in each face's vertexColors array.
	// var numberOfSides, vertexIndex
	// for ( var i = 0; i < geometry.index.length; i++ )
	// {
	// 	face = geometry.index[ i ];
	// 	numberOfSides = ( face instanceof THREE.Face3 ) ? 3 : 4;
	// 	for( var j = 0; j < numberOfSides; j++ )
	// 	{
	// 		vertexIndex = face[ faceIndices[ j ] ];
	// 		face.vertexColors[ j ].setHex(color.setHex( Math.random() * 0xffffff )) // cubeGeometry.colors[ vertexIndex ];
	// 	}
	// }
}

loadingModelPromise.then(() => {
	console.log("Loaded ONNX model.")
	if (!obj_loaded){
    	console.log("OBJ not loaded - waiting...")}
	while(!obj_loaded){}
	console.log("OBJ loaded!")
	setupModel()

})