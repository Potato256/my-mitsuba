#include <mitsuba/render/scene.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/render/renderqueue.h>
#include "myBDPT.h"

#if defined(MTS_OPENMP)
# include <omp.h>
#endif

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
    
    ref<Bitmap> m_bitmap;

    bool m_running;

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
        m_LVCMaxVertexSize = m_LVCPathSize * m_maxDepthLight;
        
        m_LVC = new BDPTVertex[m_LVCMaxVertexSize];

        m_LVCVertexSize = 0;
        m_running = true;
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
        oss<<"LVCVertexSize: "<<m_LVCVertexSize<<endl;
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
        m_LVCVertexSize = 0;



        return true;
    }

    void cancel(){
        m_running = false;
    }
    
    bool render(Scene *scene, RenderQueue *queue,
        const RenderJob *job, int sceneResID, int sensorResID, int unused) {

        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = scene->getSensor();
        ref<Film> film = sensor->getFilm();
        Vector2i cropSize = film->getCropSize();
        ref<Sampler> sampler = scene->getSampler();
        size_t sampleCount = sampler->getSampleCount();
        size_t nCores = sched->getCoreCount();
        
        int blockSize = scene->getBlockSize();

        Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " %s, " SSE_STR ") ..",
            film->getCropSize().x, film->getCropSize().y,
            nCores, nCores == 1 ? "core" : "cores");
            
        /* Create a sampler instance for every core */
        std::vector<SerializableObject *> samplers(sched->getCoreCount());
        for (size_t i=0; i<sched->getCoreCount(); ++i) {
            ref<Sampler> clonedSampler = sampler->clone();
            clonedSampler->incRef();
            samplers[i] = clonedSampler.get();
        }

        int samplerResID = sched->registerMultiResource(samplers);

        /* Allocate memory */
        BDPTVertex* tmp_LVC = new BDPTVertex[m_LVCMaxVertexSize];
        
        m_bitmap = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, cropSize);
        m_bitmap->clear();

        int* LVCSize = new int[nCores];
        memset(LVCSize, 0, sizeof(int)*nCores);

        #define MTS_OPENMP
        #if defined(MTS_OPENMP)
            #pragma omp parallel
            {   
                int tid = mts_omp_get_thread_num();
                Sampler *sampler = static_cast<Sampler *>(samplers[tid]);
                int pSize = m_LVCPathSize / nCores;
                int vSize = pSize * m_maxDepthLight * tid;
                for (int i = 0; i < pSize; ++i)
                    traceLightSubpath(sampler, scene, 
                        tmp_LVC + vSize , LVCSize[tid]);
                #pragma omp critical
                {
                    int start = m_LVCVertexSize;
                    m_LVCVertexSize += LVCSize[tid];
                    memcpy(m_LVC + start, tmp_LVC + vSize, 
                        sizeof(BDPTVertex) * LVCSize[tid]); 
                }
            }
        #else
            for (int i = 0; i < m_LVCPathSize; ++i)
                traceLightSubpath(sampler, scene, m_LVC, m_LVCVertexSize);
        #endif

        Spectrum *target = (Spectrum *) m_bitmap->getUInt8Data();

        for (int i =0; i < sampleCount; ++i) {
            if (!m_running) 
                break;
            for (int yofs=0; yofs<i; ++yofs) {
                for (int xofs=0; xofs<cropSize.x; ++xofs) {
                    target[yofs*cropSize.x+xofs] = Spectrum(1.0f);
                }
            }
            film->setBitmap(m_bitmap);       
            queue->signalRefresh(job); 
        }

        printInfos();
        
        delete []LVCSize;
        delete []tmp_LVC;
        
        return true;
    }

    void traceLightSubpath(
        Sampler* sampler, 
        const Scene* scene,
        BDPTVertex* LVC,
        int& LVCsize
    ) {
        /* Trace light subpath */
        /* Sample light position */
        PositionSamplingRecord pRec(0.0);
        Spectrum lightValue = scene->sampleEmitterPosition(pRec, sampler->next2D());  
        
        const Emitter *emitter = static_cast<const Emitter *>(pRec.object); 
        if (pRec.measure != EArea || !emitter->isOnSurface())
            SLog(EError, "myBDPT only supports area emitters!");
        
        BDPTVertex *lp = &LVC[LVCsize++];
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
            BDPTVertex *lp = &LVC[LVCsize++];
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
            BDPTVertex *last = &LVC[LVCsize-2];
            Float distSquared = its.t * its.t;
            last->pdfInverse = lastPdf * absDot(lp->wi, lp->n) / distSquared;

            if (lightTraceDepth >= 2) {
                BDPTVertex *llast = &LVC[LVCsize-3];;                
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