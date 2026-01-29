const container = document.getElementById("viewer");
const statusEl = document.getElementById("status");

function setStatus(message) {
  statusEl.textContent = message;
}

function getSceneUrl() {
  const params = new URLSearchParams(window.location.search);
  const scene = params.get("scene");
  if (!scene) {
    setStatus("No scene URL provided.");
    return null;
  }
  return scene;
}

async function loadSplat(sceneUrl) {
  setStatus("Loading Gaussian Splat…");
  try {
    const module = await import(
      "https://unpkg.com/gaussian-splats-3d@0.4.2/dist/gaussian-splats-3d.esm.js"
    );
    const viewer = new module.GaussianSplats3D.Viewer({
      rootElement: container,
      cameraUp: [0, -1, 0],
      cameraPos: [0, 0, -2],
      cameraLookAt: [0, 0, 0],
    });
    await viewer.loadFile(sceneUrl);
    viewer.start();
    setStatus("Ready (WASD / mouse)");
  } catch (error) {
    console.error(error);
    setStatus("Failed to load splat viewer. Check network access to CDN.");
  }
}

async function loadPly(sceneUrl) {
  setStatus("Loading point cloud…");
  try {
    const THREE = await import("https://unpkg.com/three@0.160.0/build/three.module.js");
    const { OrbitControls } = await import(
      "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js"
    );
    const { PLYLoader } = await import(
      "https://unpkg.com/three@0.160.0/examples/jsm/loaders/PLYLoader.js"
    );

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0f19);

    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 2);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    const loader = new PLYLoader();
    loader.load(sceneUrl, (geometry) => {
      geometry.computeVertexNormals();
      const material = new THREE.PointsMaterial({ size: 0.003, vertexColors: true });
      const mesh = new THREE.Points(geometry, material);
      scene.add(mesh);
      setStatus("Ready (orbit controls)");
    });

    function onResize() {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }
    window.addEventListener("resize", onResize);

    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();
  } catch (error) {
    console.error(error);
    setStatus("Failed to load Three.js viewer.");
  }
}

const sceneUrl = getSceneUrl();
if (sceneUrl) {
  if (sceneUrl.toLowerCase().endsWith(".ply")) {
    loadPly(sceneUrl);
  } else {
    loadSplat(sceneUrl);
  }
}
