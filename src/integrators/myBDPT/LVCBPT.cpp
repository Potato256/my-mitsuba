#include <mitsuba/render/scene.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/render/renderqueue.h>
#include <iostream>
#include <fstream>

#include <random>
#include <functional>

#include "myBDPT.h"

#if defined(MTS_OPENMP)
# include <omp.h>
#endif

#define ONLY_PT
#define ONLY_NEE
#define ONLY_BSDF
#define ONLY_BDPT
#define DEBUG

#undef ONLY_PT
#undef ONLY_NEE
#undef ONLY_BSDF
#undef ONLY_BDPT
#undef DEBUG

MTS_NAMESPACE_BEGIN

class LVCBPTIntegrator : public Integrator {
public:
    MTS_DECLARE_CLASS()

private:    
    enum MISMode {
        UniformHeuristic = 0,
        BalanceHeuristic,
        PowerHeuristic
    };    
    int m_maxDepthEye;
    int m_maxDepthLight;
    Float m_rrEye;
    Float m_rrLight;

    bool m_usePT;

    /* How many paths are there in LVC */
    int m_LVCPathSize;
    int m_LVCVertexSize;
    int m_LVCMaxVertexSize;
    int m_LVCTotalSize;
    int m_LVCConnectTimes;
    Float m_k;
    BDPTVertex* m_LVC; 
    
    ref<Bitmap> m_bitmap;
    std::vector<Point2i> blockOfs;
    int m_blockSize;
    bool m_jitterSample;
    bool m_running;
    bool m_drawCurve;    
    
    std::string m_MISmodeString;
    MISMode m_MISmode;

public:
    /// Initialize the integrator with the specified properties
    LVCBPTIntegrator(const Properties &props) : Integrator(props) {
        m_LVCPathSize = props.getInteger("LVCPathSize", 10000);
        m_maxDepthEye = props.getInteger("maxDepthEye", 50);
        m_maxDepthLight = props.getInteger("maxDepthLight", 5);
        m_rrEye = props.getFloat("rrEye", 0.75);
        m_rrLight = props.getFloat("rrLight", 0.5);
        m_blockSize = props.getInteger("blockSize", 64);
        m_jitterSample = props.getBoolean("jitterSample", true);
        m_usePT = props.getBoolean("usePT", true);
        m_LVCConnectTimes = props.getInteger("LVCConnectTimes", 10);

        m_LVCTotalSize = m_LVCPathSize * m_maxDepthLight * sizeof(BDPTVertex);
        m_LVCMaxVertexSize = m_LVCPathSize * m_maxDepthLight;
        
        m_LVC = new BDPTVertex[m_LVCMaxVertexSize];

        m_LVCVertexSize = 0;
        m_running = true;
        m_drawCurve = false;

        m_MISmodeString = props.getString("MISmode", "balance");
        if (m_MISmodeString == "uniform")
            m_MISmode = UniformHeuristic;
        else if (m_MISmodeString == "balance")
            m_MISmode = BalanceHeuristic;
        else if (m_MISmodeString == "power")
            m_MISmode = PowerHeuristic;
        else
            Log(EError, "Unknown MIS mode: %s", m_MISmodeString.c_str());
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

        blockOfs.clear();
        const Film* film = scene->getSensor()->getFilm();  
        Vector2i cropSize = film->getCropSize();
        Point2i cropOffset = film->getCropOffset();
        int w = cropSize.x / m_blockSize + 1;
        int h = cropSize.y / m_blockSize + 1;
        for (int i = 0; i < w; ++i){
            for (int j = 0; j < h; ++j){
                blockOfs.push_back(
                    Point2i(cropOffset.x+i*m_blockSize, cropOffset.y+j*m_blockSize));
            }
        }

        return true;
    }

    void cancel(){
        m_running = false;
    }
    
    bool render(Scene *scene, RenderQueue *queue,
        const RenderJob *job, int sceneResID, int sensorResID, int unused) {

        ref<Scheduler> sched = Scheduler::getInstance();
        int blockSize = scene->getBlockSize();
        
        ref<Sampler> sampler = scene->getSampler();        
        size_t sampleCount = sampler->getSampleCount();

        ref<Sensor> sensor = scene->getSensor();
        bool needsApertureSample = sensor->needsApertureSample();
        bool needsTimeSample = sensor->needsTimeSample();

        ref<Film> film = sensor->getFilm();
        Vector2i cropSize = film->getCropSize();
        Point2i cropOffset = film->getCropOffset();

        size_t nCores = sched->getCoreCount();
                
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

        Spectrum *target = (Spectrum *) m_bitmap->getUInt8Data();        
        std::string convergeCurve = "";
        m_drawCurve = false;
        if (cropSize.x == 1 && cropSize.y == 1)
            m_drawCurve = true;

        for (int i = 0; i < sampleCount; ++i) {
            if (m_drawCurve){
                if (i % 1000 == 0)
                    SLog(EInfo, "Frame: %i", i);
            }
            else
                SLog(EInfo, "Frame: %i", i);

            if (!m_running) 
                break;
            /* Trace LVC */
            m_LVCVertexSize = 0;
            memset(LVCSize, 0, sizeof(int)*nCores);
            #if defined(MTS_OPENMP)
            if (m_LVCPathSize >= nCores) {
                #pragma omp parallel
                {   
                    int tid = mts_omp_get_thread_num();
                    Sampler *sampler = static_cast<Sampler *>(samplers[tid]);
                    int pSize = m_LVCPathSize / nCores + 1;
                    int vSize = pSize * m_maxDepthLight * tid;
                    for (int i = 0; i < pSize; ++i){
                        if (i + pSize * tid >= m_LVCPathSize)
                            break;
                        traceLightSubpath(sampler, scene, 
                            tmp_LVC + vSize , LVCSize[tid]);
                    }
                    #pragma omp critical
                    {
                        int start = m_LVCVertexSize;
                        m_LVCVertexSize += LVCSize[tid];
                        memcpy(m_LVC + start, tmp_LVC + vSize, 
                            sizeof(BDPTVertex) * LVCSize[tid]); 
                    }
                }
            } else { 
                for (int i = 0; i < m_LVCPathSize; ++i)
                    traceLightSubpath(sampler, scene, m_LVC, m_LVCVertexSize);
            }
            #else
                for (int i = 0; i < m_LVCPathSize; ++i)
                    traceLightSubpath(sampler, scene, m_LVC, m_LVCVertexSize);
            #endif
            
            m_k = float(m_LVCVertexSize) / float(m_LVCPathSize);
            #if defined(ONLY_NEE) || defined(ONLY_PT)
                m_k = 1; 
            #endif

            if (!m_drawCurve)
                SLog(EInfo, "k: %f", m_k);

            /* Trace eye subpath*/
            int blockCnt = (int) blockOfs.size();
            #if defined(MTS_OPENMP)
                #pragma omp parallel for schedule(dynamic)
            #endif
            for (int block = 0; block < blockCnt; ++block) {
                int tid = mts_omp_get_thread_num();
                Sampler* sampler = static_cast<Sampler*>(samplers[tid]);
                Point2i& bOfs = blockOfs[block];
                int xBlockOfs = bOfs.x;
                int yBlockOfs = bOfs.y;

                for (int yofs=0; yofs<m_blockSize; ++yofs) {
                    for (int xofs=0; xofs<m_blockSize; ++xofs) {
                        int xRealOfs = xBlockOfs + xofs;
                        int yRealOfs = yBlockOfs + yofs;
                        if (xRealOfs >= cropSize.x || yRealOfs >= cropSize.y)
                            continue;
                        Point2 apertureSample, samplePos;
                        
                        if (needsApertureSample)
                            apertureSample = sampler->next2D();
                        samplePos = Point2(0.5, 0.5);
                        if (m_jitterSample)
                            samplePos = sampler->next2D();
                        
                        samplePos.y += cropOffset.y + yRealOfs;
                        samplePos.x += cropOffset.x + xRealOfs;
    
                        RayDifferential eyeRay;
                        Spectrum sampleValue = sensor->sampleRay(
                            eyeRay, samplePos, apertureSample, 0.0f);   
                        
                        int ofs = yRealOfs*cropSize.x+xRealOfs;
                        
                        Spectrum L = Li(eyeRay, scene, sampler);
                        float i_ = (float) i;
                        target[ofs] = (target[ofs]*i_ + L)/(i_+1.0f);
                    }
                }
            }
            film->setBitmap(m_bitmap);       
            queue->signalRefresh(job);
            if (m_drawCurve)
                convergeCurve += target[0].toString() + "\n"; 
        }

        delete []LVCSize;
        delete []tmp_LVC;
        
        printInfos();
        if (m_drawCurve){
            std::ofstream fout;
            std::ostringstream save;
            save << "./experiments/results/";
            save << (m_jitterSample ? "jitter/" : "nojitter/");
            save << toString();
            float e = round(log10(sampleCount));
            save << "-" << round(sampleCount/pow(10,e));
            save << "e" << round(log10(sampleCount));
            save << "x" << m_LVCConnectTimes << ".txt";
            fout.open(save.str().c_str());
            fout << convergeCurve;
            fout.close();
            SLog(EInfo, "Saving result to %s", save.str().c_str());
        }
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
        lp->depth = 0;
        
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
            lp->depth = lightTraceDepth;

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

    void traceEyeSubpath(
        RayDifferential &eyeRay,
        Intersection &its,
        Sampler* sampler, 
        const Scene* scene,
        std::vector<BDPTVertex*> &eyePath,
        Spectrum& Li
    ) const {
        /* Initialize throughput */
        Spectrum throughput(1.0f);
        Float lastPdf = 1.0f;
        int eyeTraceDepth = 1;
        while(eyeTraceDepth <= m_maxDepthEye) {
            /* Compute last vertex's pdfInverse */
            if (eyeTraceDepth >= 3) {
                BDPTVertex *last = eyePath[eyeTraceDepth-2];
                BDPTVertex *llast = eyePath[eyeTraceDepth-3];
                last->pdfInverse = computePdfForward(eyeRay.d, last, llast);
            }
            /* Add a vertex to the eye path */
            BDPTVertex *ep = new BDPTVertex();
            ep->pos = its.p;
            ep->wi = -eyeRay.d;
            ep->n = its.shFrame.n;
            ep->value = throughput;
            ep->uv = its.uv;
            ep->e = nullptr;
            /* Get BSDF */
            const BSDF *bsdf = its.getBSDF(eyeRay);
            ep->bsdf = bsdf;
            /* Compute pdf in area measure */
            if (eyeTraceDepth >= 2)
                ep->pdf = lastPdf * absDot(ep->wi, ep->n) / (its.t * its.t);
            eyePath.push_back(ep);
            /* Russian roulette */
            Float q = m_rrEye;
            if (sampler->next1D() >= q)
                break;
            throughput /= q;

            /* Sample BSDF * cos(theta) */
            BSDFSamplingRecord bRec(its, sampler, ERadiance);
            Spectrum bsdfWeight = bsdf->sample(bRec, lastPdf, sampler->next2D());
            const Vector3 wo = its.toWorld(bRec.wo);

            /* Update throughput */
            throughput *= bsdfWeight;
            
            /* Prepare next ray in this direction */
            eyeRay = Ray(its.p, wo, eyeRay.time);

            /* Ray intersection */
            scene->rayIntersect(eyeRay, its);
            
            /* Stop when hit nothing */
            if (!its.isValid())
                break;
            /* Compute PT contribution */
            if (its.isEmitter()) {
#ifndef ONLY_NEE
                if (m_usePT) {
                    /* Compute last vertex's inverse pdf */
                    BDPTVertex *last = eyePath[eyeTraceDepth-1];
                    if (eyeTraceDepth >= 2) {
                        BDPTVertex *llast = eyePath[eyeTraceDepth-2];
                        last->pdfInverse = computePdfForward(eyeRay.d, last, llast);
                    }
                    BDPTVertex *e = new BDPTVertex();
                    e->pos = its.p;
                    e->n = its.shFrame.n;
                    e->wi = -eyeRay.d;                    
                    e->e = its.shape->getEmitter();
                                        
                    e->pdf = lastPdf * absDot(e->n, e->wi) / (its.t * its.t);
                    e->pdfInverse = computePdfLightDir(e, last);

                    PositionSamplingRecord pRec(0.0f);
                    e->pdfLight = e->e->pdfPosition(pRec);
                    eyePath.push_back(e);

                    Li += its.Le(-eyeRay.d) * throughput * 
                        MISweight(eyePath, eyeTraceDepth, -1);
                } 
#endif
                break;
            }
            ++eyeTraceDepth;
        }
        return;
    } 

    /// Query for an unbiased estimate of the radiance along <tt>r</tt>
    Spectrum Li(const RayDifferential &r, Scene* scene, Sampler* sampler) const {
        /* Trace first eye vertex */
        Spectrum Li(0.0f);
        Intersection its;
        RayDifferential eyeRay(r);
        scene->rayIntersect(eyeRay, its);
        eyeRay.mint = Epsilon;
        
        /* Hit nothing */
        if (!its.isValid())
            return Li;        
        /* Hit emitter */
        if (its.isEmitter())
            return its.Le(-eyeRay.d);

        std::vector<BDPTVertex*> eyePath;
        /* Trace eye subpath */
        traceEyeSubpath(eyeRay, its, sampler, scene, eyePath, Li);
#ifndef ONLY_BSDF 
        /* Connect eye subpath and light subpath */
        if (eyePath.size() > 0)
            connectSubpaths(eyePath, scene, sampler, Li);
#endif
        /* Clean up */
        for (auto i : eyePath) delete i;
        return Li;
    }

    Float MISweight(
        const std::vector<BDPTVertex*> eyePath, 
        const int eyeEnd, 
        const int lightEnd
    ) const {

#ifdef ONLY_BSDF
        return 1.0f;
#endif
#ifdef ONLY_NEE
        return 1.0f;
#endif
    
        int lDepth = -1;
        int lStart = -1;
        if (lightEnd != -1) {
            lDepth = m_LVC[lightEnd].depth;
            lStart = lightEnd - lDepth;
        }
        int numStrategy = eyeEnd + lDepth + 1 + m_usePT;  
        
#ifdef ONLT_PT
        if (lDepth != 0 && lDepth != -1)
            return 0.0f;
#endif
        /**
         *              p_el     p_ll
         *  e1  ->  e2  --->  l2  ->  l1
         *  e1  <-  e2  <---  l2  <-  l1
         *      p_ee    p_le
         */
        std::vector<Float> pdfForward;
        std::vector<Float> pdfInverse;
        pdfForward.reserve(numStrategy);
        pdfInverse.reserve(numStrategy);
        Float p_ee, p_el, p_le, p_ll;

        if (lDepth == -1){
            /* PT */
            for (int i = 1; i <= eyeEnd; ++i){
                pdfForward.push_back(eyePath[i]->pdf);
                pdfInverse.push_back(eyePath[i]->pdfInverse);
            }
            pdfInverse.push_back(eyePath[eyeEnd]->pdfLight);
        }
        else if (eyeEnd == 0 && lDepth == 0){
            /* One eye vertex and one light vertex */
            BDPTVertex* e = eyePath[0];
            BDPTVertex* l = &m_LVC[lStart];
            p_el = computePdfForward(e->wi, e, l);
            p_le = computePdfLightDir(l, e);
            pdfForward.push_back(p_el);
            pdfInverse.push_back(p_le);
            pdfInverse.push_back(m_LVC[lStart].pdfLight);
        }
        else if (eyeEnd == 0) {
            /* One eye vertex */
            BDPTVertex* e = eyePath[0];
            BDPTVertex* l = &m_LVC[lightEnd];
            BDPTVertex* ll = &m_LVC[lightEnd-1];
            Vector3 d = normalize(l->pos - e->pos);
            p_el = computePdfForward(e->wi, e, l);
            p_le = computePdfForward(l->wi, l, e);
            p_ll = computePdfForward(-d, l, ll);
            
            pdfForward.push_back(p_el);
            pdfInverse.push_back(p_le);
            pdfForward.push_back(p_ll);
            pdfInverse.push_back(ll->pdfInverse);

            for (int i = lightEnd-2; i >= lStart; --i) {
                pdfForward.push_back(m_LVC[i].pdf);
                pdfInverse.push_back(m_LVC[i].pdfInverse);
            }
            pdfInverse.push_back(m_LVC[lStart].pdfLight);
        }
        else if (lDepth == 0) {
            /* One light vertex */
            BDPTVertex* ee = eyePath[eyeEnd-1];
            BDPTVertex* e = eyePath[eyeEnd];
            BDPTVertex* l = &m_LVC[lightEnd];
            Vector3 d = normalize(l->pos - e->pos);

            p_ee = computePdfForward(d, e, ee);
            p_el = computePdfForward(e->wi, e, l);
            p_le = computePdfLightDir(l, e);

            for (int i = 1; i < eyeEnd; ++i){
                pdfForward.push_back(eyePath[i]->pdf);
                pdfInverse.push_back(eyePath[i]->pdfInverse);
            }
            pdfForward.push_back(e->pdf);
            pdfInverse.push_back(p_ee);
            pdfForward.push_back(p_el);
            pdfInverse.push_back(p_le);
            pdfInverse.push_back(m_LVC[lStart].pdfLight);
        }
        else {
            /* Other case */
            BDPTVertex* ee = eyePath[eyeEnd-1];
            BDPTVertex* e = eyePath[eyeEnd];
            BDPTVertex* l = &m_LVC[lightEnd];
            BDPTVertex* ll = &m_LVC[lightEnd-1];
            Vector3 d = normalize(l->pos - e->pos);

            p_ee = computePdfForward(d, e, ee);
            p_el = computePdfForward(e->wi, e, l);
            p_le = computePdfForward(l->wi, l, e);
            p_ll = computePdfForward(-d, l, ll);

            for (int i = 1; i < eyeEnd; ++i){
                pdfForward.push_back(eyePath[i]->pdf);
                pdfInverse.push_back(eyePath[i]->pdfInverse);
            }
            pdfForward.push_back(e->pdf);
            pdfInverse.push_back(p_ee);
            pdfForward.push_back(p_el);
            pdfInverse.push_back(p_le);
            pdfForward.push_back(p_ll);
            pdfInverse.push_back(ll->pdfInverse);
            for (int i = lightEnd-2; i >= lStart; --i) {
                pdfForward.push_back(m_LVC[i].pdf);
                pdfInverse.push_back(m_LVC[i].pdfInverse);
            }
            pdfInverse.push_back(m_LVC[lStart].pdfLight);
        }

        int curStrategy = eyeEnd;
        Float denominator = 1.0f;
        Float tmp = 1.0f;
        for (int i = curStrategy - 1; i >= 0 ; --i) {
            tmp *= pdfInverse[i+1] / pdfForward[i]; 
#ifdef ONLY_PT
            if (i==numStrategy-1||i==numStrategy-2)
#endif
            denominator += tmp;
        }
        tmp = 1.0f;
        for (int i = curStrategy + 1; i < numStrategy; ++i) {
            tmp *= pdfForward[i-1] / pdfInverse[i];
#ifdef ONLY_PT
                if (i==numStrategy-1||i==numStrategy-2)
#endif
            denominator += tmp;
        }
        
        Float ans = 1.0f / denominator;
#ifdef DEBUG
        if (m_LiCount < 100) {
            std::ostringstream oss;
            oss<<endl<<"Li "<<m_LiCount<<" ";
            oss<<computePathPdf(pdfForward, pdfInverse, curStrategy, numStrategy)<<" ";
            oss<<computePathPdf(eyePath, lightPath, eyeEnd, lightEnd)<<endl;
            oss<<"mis "<<ans<<" "<<computePathMIS(eyePath, lightPath, eyeEnd, lightEnd)<<endl;
            for (auto i : pdfForward)
                oss<<i<<" ";
            oss<<endl;
            for (auto i : pdfInverse)
                oss<<i<<" ";
            oss<<endl;
            SLog(EDebug, oss.str().c_str());
        }
#endif

        if (ans < 0 || isnan(ans) || isinf(ans))
            return 0.0f;
        else
            return ans;
    }

    void connectSubpaths(
        std::vector<BDPTVertex*> eyePath, 
        const Scene* scene,
        Sampler* sampler, 
        Spectrum& Li
    ) const {
        for (int i = 0; i < m_LVCConnectTimes; ++i) {
        #if defined(ONLY_BDPT)
            for (int j = 0; j < eyePath.size(); ++j) {
                for (int choice = 0; choice < m_LVCVertexSize; ++choice) {
                    BDPTVertex* eyeEnd = eyePath[j];
                    BDPTVertex* lightEnd = &m_LVC[choice];
                    Spectrum value;
                    if(evalContri(eyeEnd, lightEnd, scene, value)) {
                        Li += value * MISweight(eyePath, j, choice)
                            / float(m_LVCConnectTimes);
                    }
                }
            }
        }
        #else
            int choice = int(sampler->next1D() * m_LVCVertexSize);
            // int choice = rand() % m_LVCVertexSize;
            // std::cout<<m_LVCVertexSize<<" "<<choice<<endl;
            // std::cout<<"size"<<m_LVCVertexSize<<" choice"<<choice<<"\n";
            #if defined(ONLY_NEE) || defined(ONLY_PT)
                choice = 0; 
            #endif
            for (int j = 0; j < eyePath.size(); ++j) {
                BDPTVertex* eyeEnd = eyePath[j];
                BDPTVertex* lightEnd = &m_LVC[choice];
                Spectrum value;
                if(evalContri(eyeEnd, lightEnd, scene, value)) {
                    Li += value * MISweight(eyePath, j, choice) * m_k
                        / float(m_LVCConnectTimes);
                }
            }
        }
        #endif
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "lvc-" << m_LVCPathSize<<"x"<<m_maxDepthLight;
        #if defined(ONLY_BDPT)
            oss << "-bdpt";
        #endif
        if (!m_usePT)
            oss << "-nopt";
        return oss.str();
    } 

    ~LVCBPTIntegrator(){
        free(m_LVC);
    }
};


MTS_IMPLEMENT_CLASS_S(LVCBPTIntegrator, false, Integrator)
MTS_EXPORT_PLUGIN(LVCBPTIntegrator, "LVCBPT");
MTS_NAMESPACE_END