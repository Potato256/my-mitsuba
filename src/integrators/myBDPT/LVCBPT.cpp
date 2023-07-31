#include <mitsuba/render/scene.h>
#include <mitsuba/core/plugin.h>
#include "myBDPT.h"

MTS_NAMESPACE_BEGIN

class LVCBPTIntegrator : public Integrator {
public:
    MTS_DECLARE_CLASS()

private:    
    int m_maxDepthEye;
    int m_maxDepthLight;
    Float m_rrEye;
    Float m_rrLight;

    /* How many paths are there in LVC */
    int m_LVCPathSize;
    int m_LVCVertexSize;
    int m_LVCMaxVertexSize;
    int m_LVCTotalSize;
    BDPTVertex* m_LVC; 
    static int m_LiCount;

public:
    /// Initialize the integrator with the specified properties
    LVCBPTIntegrator(const Properties &props) : Integrator(props) {
        m_LVCPathSize = props.getInteger("LVCPathSize", 10000);
        m_maxDepthEye = props.getInteger("maxDepthEye", 50);
        m_maxDepthLight = props.getInteger("maxDepthLight", 5);
        m_rrEye = props.getFloat("rrEye", 0.6);
        m_rrLight = props.getFloat("rrLight", 0.6);

        m_LVCTotalSize = m_LVCPathSize * m_maxDepthLight * sizeof(BDPTVertex);
        m_LVC = (BDPTVertex*) malloc(m_LVCTotalSize);

        m_LVCMaxVertexSize = m_LVCPathSize * m_maxDepthLight;
        m_LVCVertexSize = 0;
    }
    
    // Unserialize from a binary data stream
    LVCBPTIntegrator(Stream *stream, InstanceManager *manager)
        : Integrator(stream, manager) {}
    
    /// Serialize to a binary data stream
    void serialize(Stream *stream, InstanceManager *manager) const {
        Integrator::serialize(stream, manager);
    }

    void printInfos(){
        std::ostringstream oss;
        oss<<"\n---------- LVCBPT Info Print -----------\n";
        oss<<"maxDepthEye: "<<m_maxDepthEye<<endl;
        oss<<"maxDepthLight: "<<m_maxDepthLight<<endl;
        oss<<"rrEye: "<<m_rrEye<<endl;
        oss<<"rrLight: "<<m_rrLight<<endl;
        // oss<<"MISmode: "<<m_MISmodeString<<endl;
        oss<<"BDPTVertex size: "<<sizeof(BDPTVertex)<<endl;
        oss<<"LVCPathSize: "<<m_LVCPathSize<<endl;
        oss<<"LVCMaxVertexSize: "<<m_LVCMaxVertexSize<<endl;
        oss<<"LVCTotalSize: "<<m_LVCMaxVertexSize/1024 <<"KB"<<endl;
        oss<<"-----------------------------------------\n";
        SLog(EDebug, oss.str().c_str());
    }

    /// Preprocess function -- called on the initiating machine
    bool preprocess(const Scene *scene, RenderQueue *queue,
        const RenderJob *job, int sceneResID, int cameraResID, 
        int samplerResID) {
        Integrator::preprocess(scene, queue, job, sceneResID,
            cameraResID, samplerResID);
        printInfos();



        return true;
    }

    void cancel(){}
    
    bool render(Scene *scene, RenderQueue *queue,
        const RenderJob *job, int sceneResID, int sensorResID, int unused) {

        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = scene->getSensor();
        ref<Film> film = sensor->getFilm();
        size_t nCores = sched->getCoreCount();
        
        Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " %s, " SSE_STR ") ..",
            film->getCropSize().x, film->getCropSize().y,
            nCores, nCores == 1 ? "core" : "cores");
            
        Vector2i cropSize = film->getCropSize();
        
        ref<Sampler> sampler = static_cast<Sampler*> (PluginManager::getInstance()->
            createObject(MTS_CLASS(Sampler), Properties("independent")));

        int blockSize = scene->getBlockSize();

        for (int i = 0; i < m_LVCPathSize; ++i)
            traceLightSubpath(sampler, scene);
        
        return true;
    }

    void traceLightSubpath(
        Sampler* sampler, 
        const Scene* scene
    ) {
        /* Trace light subpath */
        /* Sample light position */
        PositionSamplingRecord pRec(0.0);
        Spectrum lightValue = scene->sampleEmitterPosition(pRec, sampler->next2D());  
        
        const Emitter *emitter = static_cast<const Emitter *>(pRec.object); 
        if (pRec.measure != EArea || !emitter->isOnSurface())
            SLog(EError, "myBDPT only supports area emitters!");
        
        BDPTVertex *lp = &m_LVC[m_LVCVertexSize++];
        lp->pos = pRec.p;
        lp->n = pRec.n;
        lp->value = lightValue;
        lp->pdfLight = pRec.pdf;
        lp->e = emitter;
        
        /* Sample light direction */
        DirectionSamplingRecord dRec;       
        emitter->sampleDirection(dRec, pRec, sampler->next2D());
        if (dRec.measure != ESolidAngle)
            SLog(EError, "myBDPT only supports emitters with direction sampling on SolidAngle!");

        /* Trace a ray in this direction */
        RayDifferential lightRay(pRec.p, dRec.d, 0.0f);

        /* The sampling pdf of last vertex */
        Float lastPdf = dRec.pdf ;

        /* Initialize throughput */
        Spectrum throughput(lightValue * emitter->evalDirection(dRec, pRec)
            / dRec.pdf);
        
        Intersection its;
        int lightTraceDepth = 1;
        while(lightTraceDepth < m_maxDepthLight) {
            /* Russian roulette */
            Float q = m_rrLight;
            if (sampler->next1D() >= q)
                break;
            throughput /= q;

            /* Ray intersection */
            scene->rayIntersect(lightRay, its);
            /* Stop when hit nothing or hit an emitter */
            if (!its.isValid() || its.isEmitter())
                break;
            
            /* Add a vertex to the light path */
            BDPTVertex *lp = &m_LVC[m_LVCVertexSize++];
            lp->pos = its.p;
            lp->wi = -lightRay.d;
            lp->n = its.shFrame.n;
            lp->value = throughput;
            lp->uv = its.uv;
            lp->e = nullptr;
            /* Get BSDF */
            const BSDF *bsdf = its.getBSDF(lightRay);
            lp->bsdf = bsdf;

            /* The order of pdf/pdfInverse is opposite here comparing to eye trace! */
            BDPTVertex *last = &m_LVC[m_LVCVertexSize-2];
            Float distSquared = its.t * its.t;
            last->pdfInverse = lastPdf * absDot(lp->wi, lp->n) / distSquared;

            if (lightTraceDepth >= 2) {
                BDPTVertex *llast = &m_LVC[m_LVCVertexSize-3];;                
                llast->pdf = computePdfForward(lightRay.d, last, llast);
            }

            /* Sample BSDF * cos(theta) */
            BSDFSamplingRecord bRec(its, sampler, ERadiance);
            Spectrum bsdfWeight = bsdf->sample(bRec, lastPdf, sampler->next2D());
            const Vector wo = its.toWorld(bRec.wo);

            /* Update throughput */
            throughput *= bsdfWeight;
            /* Prepare next ray in this direction */
            lightRay = Ray(its.p, wo, lightRay.time);
            ++lightTraceDepth;
        }
        return;
    } 


    ~LVCBPTIntegrator(){
        free(m_LVC);
    }
};




int LVCBPTIntegrator::m_LiCount = 0;

MTS_IMPLEMENT_CLASS_S(LVCBPTIntegrator, false, Integrator)
MTS_EXPORT_PLUGIN(LVCBPTIntegrator, "LVCBPT");
MTS_NAMESPACE_END