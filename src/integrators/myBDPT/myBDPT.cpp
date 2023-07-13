#include <mitsuba/render/scene.h>
#include "myBDPT.h"

MTS_NAMESPACE_BEGIN

class myBDPTIntegrator : public SamplingIntegrator {
public:
    MTS_DECLARE_CLASS()

private:
    int m_maxDepthEye;
    int m_maxDepthLight;
    Float m_rrEye;
    Float m_rrLight;
    static int m_LiCount;

public:
    /// Initialize the integrator with the specified properties
    myBDPTIntegrator(const Properties &props) : SamplingIntegrator(props) {
        m_maxDepthEye = props.getInteger("maxDepthForEye", 20);
        m_maxDepthLight = props.getInteger("maxDepthForLight", 20);
        m_rrEye = props.getFloat("rrEye", 0.5);
        m_rrLight = props.getFloat("rrLight", 0.5);
    }
    // Unserialize from a binary data stream
    myBDPTIntegrator(Stream *stream, InstanceManager *manager)
        : SamplingIntegrator(stream, manager) {}
    
    /// Serialize to a binary data stream
    void serialize(Stream *stream, InstanceManager *manager) const {
        SamplingIntegrator::serialize(stream, manager);
    }

    /// Preprocess function -- called on the initiating machine
    bool preprocess(const Scene *scene, RenderQueue *queue,
        const RenderJob *job, int sceneResID, int cameraResID, 
        int samplerResID) {
        SamplingIntegrator::preprocess(scene, queue, job, sceneResID,
            cameraResID, samplerResID);
        return true;
    }

    /// Query for an unbiased estimate of the radiance along <tt>r</tt>
    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        ++m_LiCount;
        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        Sampler *sampler = rRec.sampler;
        RayDifferential eyeRay(r);
        Spectrum Li(0.0f);
        Spectrum throughput(1.0f);
        Float eta = 1.0f;
        
        std::vector<BDPTVertex*> lightPath;
        std::vector<BDPTVertex*> eyePath;

        /* Trace first eye vertex */
        rRec.rayIntersect(eyeRay);
        eyeRay.mint = Epsilon;
        
        /* Hit nothing */
        if (!its.isValid())
            return Li;
        
        /* Hit emitter */
        if ( its.isEmitter())
            return its.Le(-eyeRay.d);

        /* Trace eye subpath */
        traceEyeSubpath(eyeRay, its, sampler, scene, eyePath, Li);

        /* Trace light subpath */
        traceLightSubpath(its, sampler, scene, lightPath);

        /* Connect eye subpath and light subpath */
        connectSubpaths(eyePath, lightPath, scene, Li);

        /* Clean up */
        for (int i = 0; i < lightPath.size(); ++i)
            delete lightPath[i];
        for (int i = 0; i < eyePath.size(); ++i)
            delete eyePath[i];

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

        /* The sampling pdf of last vertex */
        // Float lastPdf = 1.0f;
        /* Initialize throughput */
        Spectrum throughput(1.0f);
        Float eta = 1.0f;

        int eyeTraceDepth = 1;
        while(eyeTraceDepth <= m_maxDepthEye || m_maxDepthEye < 0) {

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
            ep->pdf = 1.0f;
            eyePath.push_back(ep);

            /* Russian roulette */
            Float q = m_rrEye;
            if (sampler->next1D() >= q)
                break;
            throughput /= q;
            // lastPdf *= q;

            /* Sample BSDF * cos(theta) */
            Float bsdfPdf;
            BSDFSamplingRecord bRec(its, sampler, ERadiance);
            Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, sampler->next2D());
            // Spectrum bsdfVal = bsdf->eval(bRec) / Frame::cosTheta(bRec.wo);
            const Vector wo = its.toWorld(bRec.wo);

            /* Update throughput */
            throughput *= bsdfWeight;
            
            /* Compute geometric term */
            // Float G = abs(its.wi.z) / its.p.distanceSquared(lightRay.p);
            
            /* Compute its pdf on area measure */
            // nextlp->pdf = lp->pdf * lastPdf * G;
            
            /* Update lastPdf */
            // lastPdf = bsdfPdf;
            
            /* Update lastCosine */
            // lastCosine = abs(bRec.wo.z);
            
            /* Update eta */
            eta *= bRec.eta;
            
            /* Prepare next ray in this direction */
            eyeRay = Ray(its.p, wo, eyeRay.time);

            /* Ray intersection */
            scene->rayIntersect(eyeRay, its);
            
            /* Stop when hit nothing */
            if (!its.isValid())
                break;
            /* Compute PT contribution */
            if (its.isEmitter()) {
                Li += its.Le(-eyeRay.d) * throughput / (eyeTraceDepth);
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
        lp->pdf = pRec.pdf;
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
        /* The cosine term of last vertex */
        Float lastCosine = absDot(dRec.d, pRec.n);

        /* Initialize throughput */
        Spectrum throughput(lightValue * emitter->evalDirection(dRec, pRec)
            / dRec.pdf);
        Float eta = 1.0f;

        int lightTraceDepth = 1;
        while(lightTraceDepth <= m_maxDepthLight || m_maxDepthLight < 0) {

            /* Russian roulette */
            Float q = m_rrLight;
            if (sampler->next1D() >= q)
                break;
            throughput /= q;
            lastPdf *= q;

            /* Ray intersection */
            scene->rayIntersect(lightRay, its);

            /* Stop when hit nothing or hit an emitter */
            if (!its.isValid() || its.isEmitter())
                break;
            
            /* Add a vertex to the light path */
            BDPTVertex *nextlp = new BDPTVertex();
            nextlp->pos = its.p;
            nextlp->wi = -lightRay.d;
            nextlp->n = its.shFrame.n;
            nextlp->value = throughput;
            nextlp->uv = its.uv;
            nextlp->e = nullptr;
            /* Get BSDF */
            const BSDF *bsdf = its.getBSDF(lightRay);
            nextlp->bsdf = bsdf;   
            lightPath.push_back(nextlp);
            
            /* Sample BSDF * cos(theta) */
            Float bsdfPdf;
            BSDFSamplingRecord bRec(its, sampler, ERadiance);
            Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf,  sampler->next2D());
            // Spectrum bsdfVal = bsdf->eval(bRec);
            const Vector wo = its.toWorld(bRec.wo);
            
            /* Compute geometric term */
            Float G = abs(its.wi.z) / (its.p - lightRay.o).lengthSquared();
            
            /* Compute its pdf on area measure */
            // nextlp->pdf = lp->pdf * lastPdf * G;
            
            /* Update throughput */
            throughput *= bsdfWeight;

            /* Update lp */
            lp = nextlp;

            /* Update lastPdf */
            lastPdf = bsdfPdf;
            
            /* Update lastCosine */
            lastCosine = abs(bRec.wo.z);
            
            /* Update eta */
            eta *= bRec.eta;
            
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
                if (eyeEnd->value.max() <= 0 || lightEnd->value.max() <=0)
                    continue;

                Vector d = lightEnd->pos - eyeEnd->pos;
                Float distSquared = d.lengthSquared();
                Float dist = std::sqrt(distSquared);
                d /= dist;

                Ray ray(eyeEnd->pos, d, Epsilon,
                            dist*(1-ShadowEpsilon), 0.0f);
                if (scene->rayIntersect(ray))
                    continue;

                Intersection its;
                its.p = eyeEnd->pos;
                its.shFrame = Frame(eyeEnd->n);
                its.uv = eyeEnd->uv;
                its.hasUVPartials = 0;

                /* Allocate a record for querying the BSDF */
                BSDFSamplingRecord bRec(its, its.toLocal(eyeEnd->wi), its.toLocal(d), ERadiance);
                Spectrum bsdfVal1 = eyeEnd->bsdf->eval(bRec);

                if (lightEnd->e != nullptr) {
                    DirectionSamplingRecord dRec(-d);
                    PositionSamplingRecord pRec(0.0f);
                    pRec.n = lightEnd->n;
                    Li += eyeEnd->value * lightEnd->value * 
                        lightEnd->e->evalDirection(dRec, pRec) * bsdfVal1 
                        / distSquared / (i+j+2);
                } else {
                    its.p = lightEnd->pos;
                    its.shFrame = Frame(lightEnd->n);
                    its.uv = lightEnd->uv;
                    BSDFSamplingRecord bRec(its, its.toLocal(lightEnd->wi), its.toLocal(-d),ERadiance);

                    Spectrum bsdfVal2 = lightEnd->bsdf->eval(bRec);
                    Li += eyeEnd->value * lightEnd->value * bsdfVal1 * bsdfVal2
                        / distSquared / (i+j+2);
                }
            }
        }
        return;
    }

public:

};

int myBDPTIntegrator::m_LiCount = 0;

MTS_IMPLEMENT_CLASS_S(myBDPTIntegrator, false, SamplingIntegrator)
MTS_EXPORT_PLUGIN(myBDPTIntegrator, "My BDPT integrator");
MTS_NAMESPACE_END