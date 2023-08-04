#include <mitsuba/render/scene.h>
#include <mitsuba/core/plugin.h>
#include "myBDPT.h"

#define ONLY_PT
#define DEBUG

// #undef ONLY_PT
#undef DEBUG

MTS_NAMESPACE_BEGIN

class myBDPTIntegrator : public SamplingIntegrator {
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
    static int m_LiCount;
    static Float m_lightLength;
    static int m_lightCnt;

    std::string m_MISmodeString;
    MISMode m_MISmode;

public:
    /// Initialize the integrator with the specified properties
    myBDPTIntegrator(const Properties &props) : SamplingIntegrator(props) {
        m_maxDepthEye = props.getInteger("maxDepthEye", 50);
        m_maxDepthLight = props.getInteger("maxDepthLight", 50);
        m_rrEye = props.getFloat("rrEye", 0.6);
        m_rrLight = props.getFloat("rrLight", 0.6);
        m_usePT = props.getBoolean("usePT", true);

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
    myBDPTIntegrator(Stream *stream, InstanceManager *manager)
        : SamplingIntegrator(stream, manager) {}
    
    /// Serialize to a binary data stream
    void serialize(Stream *stream, InstanceManager *manager) const {
        SamplingIntegrator::serialize(stream, manager);
    }

    void printInfos(){
        std::ostringstream oss;
        oss<<"\n------------ BDPT Info Print ------------\n";
        oss<<"maxDepthEye: "<<m_maxDepthEye<<endl;
        oss<<"maxDepthLight: "<<m_maxDepthLight<<endl;
        oss<<"rrEye: "<<m_rrEye<<endl;
        oss<<"rrLight: "<<m_rrLight<<endl;
        oss<<"MISmode: "<<m_MISmodeString<<endl;
        oss<<"usePT: "<<m_usePT<<endl;
        oss<<"BDPTVertex size: "<<sizeof(BDPTVertex)<<endl;
        oss<<"lightCnt: "<<m_lightCnt<<endl;
        oss<<"lightLength: "<<m_lightLength<<endl;
        oss<<"-----------------------------------------\n";
        SLog(EDebug, oss.str().c_str());
    }

    /// Preprocess function -- called on the initiating machine
    bool preprocess(const Scene *scene, RenderQueue *queue,
        const RenderJob *job, int sceneResID, int cameraResID, 
        int samplerResID) {
        SamplingIntegrator::preprocess(scene, queue, job, sceneResID,
            cameraResID, samplerResID);
        
        ref<Sampler> sampler = static_cast<Sampler *> (PluginManager::getInstance()->
            createObject(MTS_CLASS(Sampler), Properties("independent")));

            for (int i = 0; i < 1000000; ++i) {
                std::vector<BDPTVertex*> lightPath;
            Intersection its;
            traceLightSubpath(its, sampler, scene, lightPath);
            if (m_lightCnt == 0)
                m_lightLength = lightPath.size();
            else
                m_lightLength += (lightPath.size() - m_lightLength) / (m_lightCnt + 1);
            ++m_lightCnt;
            for (auto i : lightPath) delete i;
        }
        
        printInfos();
        return true;
    }

    /// Query for an unbiased estimate of the radiance along <tt>r</tt>
    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        ++m_LiCount;

        Spectrum Li(0.0f);
        /* Trace first eye vertex */
        RayDifferential eyeRay(r);
        rRec.rayIntersect(eyeRay);
        eyeRay.mint = Epsilon;
        
        /* Hit nothing */
        if (!rRec.its.isValid())
            return Li;        
        /* Hit emitter */
        if ( rRec.its.isEmitter())
            return rRec.its.Le(-eyeRay.d);

        std::vector<BDPTVertex*> eyePath;
        /* Trace eye subpath */
        traceEyeSubpath(eyeRay, rRec.its, rRec.sampler, rRec.scene, eyePath, Li);

        std::vector<BDPTVertex*> lightPath;
        /* Trace light subpath */
        traceLightSubpath(rRec.its, rRec.sampler, rRec.scene, lightPath);

        /* Connect eye subpath and light subpath */
        connectSubpaths(eyePath, lightPath, rRec.scene, Li);

        /* Clean up */
        for (auto i : eyePath)   delete i;
        for (auto i : lightPath) delete i;

        return Li;
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
        while(eyeTraceDepth <= m_maxDepthEye || m_maxDepthEye < 0) {
            
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
                    /* For pdflight */
                    std::vector<BDPTVertex*> lightPath;
                    lightPath.push_back(e);
                    Li += its.Le(-eyeRay.d) * throughput * 
                        MISweight(eyePath, lightPath, eyeTraceDepth, -1);
                } 
                break;
            }
            ++eyeTraceDepth;
        }
        return;
    } 

    void traceLightSubpath(
        Intersection &its,
        Sampler* sampler, 
        const Scene* scene,
        std::vector<BDPTVertex*> &lightPath
    ) const {
        /* Trace light subpath */
        /* Sample light position */
        PositionSamplingRecord pRec(0.0);
        Spectrum lightValue = scene->sampleEmitterPosition(pRec, sampler->next2D());  
        
        const Emitter *emitter = static_cast<const Emitter *>(pRec.object); 
        if (pRec.measure != EArea || !emitter->isOnSurface())
            SLog(EError, "myBDPT only supports area emitters!");
        
        BDPTVertex *lp = new BDPTVertex();
        lp->pos = pRec.p;
        lp->n = pRec.n;
        lp->value = lightValue;
        lp->pdfLight = pRec.pdf;
        lp->e = emitter;
        lightPath.push_back(lp);
        
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

        int lightTraceDepth = 1;
        while(lightTraceDepth <= m_maxDepthLight || m_maxDepthLight < 0) {
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
            BDPTVertex *lp = new BDPTVertex();
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
            BDPTVertex *last = lightPath[lightTraceDepth-1];
            Float distSquared = its.t * its.t;
            last->pdfInverse = lastPdf * absDot(lp->wi, lp->n) / distSquared;

            if (lightTraceDepth >= 2) {
                BDPTVertex *llast = lightPath[lightTraceDepth-2];                
                llast->pdf = computePdfForward(lightRay.d, last, llast);
            }

            lightPath.push_back(lp);
            
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

    void connectSubpaths(
        std::vector<BDPTVertex*> eyePath, 
        std::vector<BDPTVertex*> lightPath,
        const Scene* scene, 
        Spectrum& Li
    ) const {
        if (eyePath.size() == 0 || lightPath.size() == 0)
            return;
        for (int i = 0; i < eyePath.size(); ++i) {
            for (int j = 0; j < lightPath.size(); ++j) {
                BDPTVertex* eyeEnd = eyePath[i];
                BDPTVertex* lightEnd = lightPath[j];
                Spectrum value;
                if(evalContri(eyeEnd, lightEnd, scene, value))
                    Li += value * MISweight(eyePath, lightPath, i, j);
            }
        }

        // int lSize = lightPath.size();
        // for (int i = 0; i < eyePath.size(); ++i) {
        //     int choice = rand() % lSize;
        //         BDPTVertex* eyeEnd = eyePath[i];
        //         BDPTVertex* lightEnd = lightPath[choice];
        //         Spectrum value;
        //         if(evalContri(eyeEnd, lightEnd, scene, value))
        //             Li += value * MISweight(eyePath, lightPath, i, choice) * lSize;
        // }
    }
    Float MISweight(
        std::vector<BDPTVertex*> eyePath, 
        std::vector<BDPTVertex*> lightPath,
        int eyeEnd, 
        int lightEnd
    ) const {

        int numStrategy = eyeEnd + lightEnd + 1 + m_usePT;
        if (m_MISmode == UniformHeuristic) 
            return 1.0f / numStrategy;        

#ifdef ONLY_PT
        if (lightEnd != 0 && lightEnd != -1)
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
        Float p_ee, p_el, p_le, p_ll;

        if (lightEnd == -1){
            /* PT */
            for (int i = 1; i <= eyeEnd; ++i){
                pdfForward.push_back(eyePath[i]->pdf);
                pdfInverse.push_back(eyePath[i]->pdfInverse);
            }
        }
        else if (eyeEnd == 0 && lightEnd == 0){
            /* One eye vertex and one light vertex */
            BDPTVertex* e = eyePath[0];
            BDPTVertex* l = lightPath[0];
            p_el = computePdfForward(e->wi, e, l);
            p_le = computePdfLightDir(l, e);
            pdfForward.push_back(p_el);
            pdfInverse.push_back(p_le);
        }
        else if (eyeEnd == 0) {
            /* One eye vertex */
            BDPTVertex* e = eyePath[0];
            BDPTVertex* l = lightPath[lightEnd];
            BDPTVertex* ll = lightPath[lightEnd-1];
            Vector3 d = normalize(l->pos - e->pos);
            p_el = computePdfForward(e->wi, e, l);
            p_le = computePdfForward(l->wi, l, e);
            p_ll = computePdfForward(-d, l, ll);
            
            pdfForward.push_back(p_el);
            pdfInverse.push_back(p_le);
            pdfForward.push_back(p_ll);
            pdfInverse.push_back(ll->pdfInverse);

            for (int i = lightEnd-2; i >= 0; --i) {
                pdfForward.push_back(lightPath[i]->pdf);
                pdfInverse.push_back(lightPath[i]->pdfInverse);
            }
        }
        else if (lightEnd == 0) {
            /* One light vertex */
            BDPTVertex* ee = eyePath[eyeEnd-1];
            BDPTVertex* e = eyePath[eyeEnd];
            BDPTVertex* l = lightPath[0];
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
        }
        else {
            /* Other case */
            BDPTVertex* ee = eyePath[eyeEnd-1];
            BDPTVertex* e = eyePath[eyeEnd];
            BDPTVertex* l = lightPath[lightEnd];
            BDPTVertex* ll = lightPath[lightEnd-1];
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
            for (int i = lightEnd-2; i >= 0; --i) {
                pdfForward.push_back(lightPath[i]->pdf);
                pdfInverse.push_back(lightPath[i]->pdfInverse);
            }
        }
        pdfInverse.push_back(lightPath[0]->pdfLight);

        int curStrategy = eyeEnd;
        Float denominator = 1.0f;

        if (m_MISmode == BalanceHeuristic) {
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
        }
        if (m_MISmode == PowerHeuristic){
            Float tmp = 1.0f;
            for (int i = curStrategy - 1; i >= 0 ; --i) {
                tmp *= pdfInverse[i+1] / pdfForward[i];
#ifdef ONLY_PT
                if (i==numStrategy-1||i==numStrategy-2)
#endif
                denominator += tmp*tmp;
            }
            tmp = 1.0f;
            for (int i = curStrategy + 1; i < numStrategy; ++i) {
                tmp *= pdfForward[i-1] / pdfInverse[i];
#ifdef ONLY_PT
                if (i==numStrategy-1||i==numStrategy-2)
#endif
                denominator += tmp*tmp;
            }
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

    Float computePathPdf(
        std::vector<Float> pdfForward,
        std::vector<Float> pdfInverse,
        int curStrategy,
        int numStrategy
    ) const {
        Float curPdf = 1.0f;
        for (int i = curStrategy - 1; i >= 0 ; --i)
            curPdf *= pdfForward[i]; 
        for (int i = curStrategy + 1; i < numStrategy; ++i)
            curPdf *= pdfInverse[i];
        return curPdf;
    }

    void checkEyepathPdf(int limit, std::vector<BDPTVertex*> eyePath) const {
        if (m_LiCount < limit && eyePath.size() > 1) {
            std::ostringstream oss;
            oss<<endl<<"Li_cnt "<<m_LiCount<<" eye path "<<endl;
            for (int i = 1; i < eyePath.size(); ++i) {
                oss<<i<<": "<<eyePath[i]->pdf-computePdfForward(eyePath[i-1]->wi, eyePath[i-1], eyePath[i])<<" ";
                if (i != eyePath.size()-1)
                    oss<<computePdfForward(normalize(eyePath[i+1]->pos-eyePath[i]->pos),
                        eyePath[i], eyePath[i-1])-eyePath[i]->pdfInverse<<endl;
                else
                    oss<<eyePath[i]->pdfInverse<<endl;
            }
            SLog(EDebug, oss.str().c_str());
        }
    }

    void checkLightpathPdf(int limit, std::vector<BDPTVertex*> lightPath) const {
        if (m_LiCount < limit && lightPath.size() > 1) {
            std::ostringstream oss;
            oss<<endl<<"Li_cnt "<<m_LiCount<<" light path "<<endl;
            for (int i = int(lightPath.size()-2); i >=0; --i) {
                oss<<i<<": ";
                if (i != lightPath.size()-2)
                    oss<<computePdfForward(normalize(lightPath[i+2]->pos-lightPath[i+1]->pos),
                        lightPath[i+1], lightPath[i])-lightPath[i]->pdf<<" ";
                else
                    oss<<lightPath[i]->pdf<<" ";

                if (i != 0)
                    oss<<computePdfForward(normalize(lightPath[i-1]->pos-lightPath[i]->pos),
                        lightPath[i], lightPath[i+1])-lightPath[i]->pdfInverse<<endl;
                else
                    oss<<computePdfLightDir(lightPath[i],lightPath[i+1])
                            -lightPath[i]->pdfInverse<<endl;
            }
            SLog(EDebug, oss.str().c_str());
        }
    }

};

int myBDPTIntegrator::m_LiCount = 0;
Float myBDPTIntegrator::m_lightLength = 0;
int myBDPTIntegrator::m_lightCnt = 0;

MTS_IMPLEMENT_CLASS_S(myBDPTIntegrator, false, SamplingIntegrator)
MTS_EXPORT_PLUGIN(myBDPTIntegrator, "My BDPT integrator");
MTS_NAMESPACE_END